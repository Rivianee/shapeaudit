"""Gera ons_cmo_se_mensal.csv a partir dos dados abertos do ONS.

Fontes (https://dados.ons.org.br):
  - CMO semanal por subsistema   (dataset cmo_se)
  - EAR diário por subsistema    (dataset ear_subsistema_di)

Uso:
    python gerar_ons_cmo_se_mensal.py              # baixa direto do ONS
    python gerar_ons_cmo_se_mensal.py <pasta>      # usa CSVs já baixados
"""
import sys
from pathlib import Path

import pandas as pd

S3 = 'https://ons-aws-prod-opendata.s3.amazonaws.com/dataset'
ANOS = range(2005, pd.Timestamp.today().year + 1)
SAIDA = Path(__file__).with_name('ons_cmo_se_mensal.csv')


def ler(pasta, dataset, nome, ano):
    arq = f'{nome}_{ano}.csv'
    fonte = Path(pasta) / arq if pasta else f'{S3}/{dataset}/{arq}'
    return pd.read_csv(fonte, sep=';')


def main(pasta=None):
    cmo = pd.concat([ler(pasta, 'cmo_se', 'CMO_SEMANAL', a) for a in ANOS])
    ear = pd.concat([ler(pasta, 'ear_subsistema_di', 'EAR_DIARIO_SUBSISTEMA', a)
                     for a in ANOS])

    cmo = cmo[cmo.id_subsistema == 'SE']
    ear = ear[ear.id_subsistema == 'SE']
    cmo['data'] = pd.to_datetime(cmo.din_instante).dt.to_period('M')
    ear['data'] = pd.to_datetime(ear.ear_data).dt.to_period('M')

    # só meses completos: o EAR precisa cobrir todos os dias do mês
    dias = ear.groupby('data').ear_data.nunique()
    completos = dias[dias == dias.index.days_in_month].index

    m = pd.DataFrame({
        'cmo': cmo.groupby('data').val_cmomediasemanal.mean(),
        'ear': ear.groupby('data').ear_verif_subsistema_percentual.mean(),
        'ear_min': ear.groupby('data').ear_verif_subsistema_percentual.min(),
    })
    m = m[m.index.isin(completos)].dropna().sort_index()

    m['cmo_lag1'] = m.cmo.shift(1)
    m['ear_lag1'] = m.ear.shift(1)
    m = m.dropna()

    m['ano'] = m.index.year
    m['mes'] = m.index.month
    m['data'] = m.index.to_timestamp().strftime('%Y-%m-%d')
    m = m[['data', 'ano', 'mes', 'cmo', 'cmo_lag1', 'ear', 'ear_min', 'ear_lag1']]
    m.round(2).to_csv(SAIDA, index=False)
    print(f'{len(m)} meses, de {m.data.min()[:7]} a {m.data.max()[:7]} -> {SAIDA}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
