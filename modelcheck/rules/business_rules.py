"""Carrega e representa regras de negócio conhecidas."""
import yaml


class BusinessRule:
    def __init__(self, feature, expected_direction, description):
        self.feature = feature
        self.expected_direction = expected_direction
        self.description = description


class RuleSet:
    def __init__(self, rules: list):
        self.rules = rules

    @classmethod
    def from_yaml(cls, path):
        with open(path) as file:
            raw_rules = yaml.safe_load(file)

        rules = []
        for item in raw_rules["rules"]:
            rule = BusinessRule(
                feature=item["feature"],
                expected_direction=item["expected_direction"],
                description=item["description"],
            )
            rules.append(rule)

        return cls(rules)

    def get_rule_for_feature(self, feature):
        for rule in self.rules:
            if rule.feature == feature:
                return rule
        return None
