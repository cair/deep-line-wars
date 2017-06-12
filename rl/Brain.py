from rl.DecisionTree import DecisionNode


class Brain:

    @staticmethod
    def def_or_attack(player, game):
        pass


    def __init__(self, player, game):
        # Root Node evaluates
        root = DecisionNode("Defense or Attack")
        attack = DecisionNode("Attack")
        attack_type = DecisionNode("Attack Unit Evaluation")

        defense = DecisionNode("Defense")
        defense_type = DecisionNode("Defense Type Evaluation")
        defense_placement = DecisionNode("Defense Placement Evaluation")

        standing_eval = DecisionNode("Game Standing Evaluation")

        # Root
        root.children.append(attack)
        root.children.append(defense)
        root.children.append(standing_eval)

        # Attack
        attack.children.append(attack_type)
        attack.children.append(standing_eval)

        # Defense
        defense.children.append(defense_type)
        defense_type.children.append(defense_placement)
        defense_type.children.append(standing_eval)

        ##########################################
        # Functions
        ##########################################
        root.f = [Brain.def_or_attack_or_noop, player, game]
