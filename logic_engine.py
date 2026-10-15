class KnowledgeBase:
    """A declarative Knowledge Base and Forward Chaining Inference Engine."""

    def __init__(self):
        # Set to store unique string facts (e.g., "TargetVisible")
        self.facts = set()
        # List to store rules as Tuples: ( [list_of_premises], "conclusion_string" )
        self.rules = []

    def tell_fact(self, fact_string: str):
        """Add a fact string to the facts set."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list: list, conclusion_string: str):
        """Append a rule as a tuple of premises and conclusion."""
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self):
        """Empty the facts set."""
        self.facts.clear()

    def forward_chain(self):
        """Data-driven forward chaining algorithm using Modus Ponens."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens Check
                    if all(premise in self.facts for premise in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True
