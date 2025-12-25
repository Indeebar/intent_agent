from rules.budget import extract_budget
from rules.category import extract_category
from rules.intent_rules import extract_intent_rule_based


def parse_user_query(text: str):
    return {
        "intent": extract_intent_rule_based(text),
        "category": extract_category(text),
        "budget": extract_budget(text)
    }


if __name__ == "__main__":
    query = "I want a budget phone under 15000 for gaming"
    result = parse_user_query(query)
    print(result)
