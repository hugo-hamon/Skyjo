import numpy as np
import random

CARD_COUNT = 12
CARD_DISTRIBUTION = {
    -2: 5,
    -1: 10,
    0: 15,
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10,
    11: 10,
    12: 10,
}


class SkyjoEnv:
    def __init__(self, num_players: int = 4) -> None:
        """
        Initialize the Skyjo environment.

        Args:
            num_players (int): Number of players in the game. Default is 4. Must be at least 2. And at most 8.
        """
        if num_players < 2 or num_players > 8:
            raise ValueError("Number of players must be between 2 and 8")
        self.num_players = num_players
        self.current_player = 0
        self.deck = []
        self.discard_pile = []
        self.players = []
        self.done = False
        self.final_player = None
        self.remaining_final_turns = None

        self.pending_card = None
        self.previous_action = ""

    def reset(self) -> dict:
        """
        Resets the environment to the initial state:
        - Shuffles the deck
        - Deals 12 cards to each player
        - Makes 2 cards visible for each player
        - Prepares the discard pile

        Returns:
            dict: The initial observation of the environment.
        """
        self.deck = []
        for value, count in CARD_DISTRIBUTION.items():
            self.deck.extend([value] * count)
        random.shuffle(self.deck)

        self.players = []
        for _ in range(self.num_players):
            grid = [self.deck.pop() for _ in range(CARD_COUNT)]
            visible = [False] * CARD_COUNT
            # Make 2 cards visible randomly
            indices = random.sample(range(CARD_COUNT), 2)
            current_score = 0
            for i in indices:
                visible[i] = True
                current_score += grid[i]

            self.players.append(
                {
                    "grid": grid,
                    "visible": visible,
                    "return_count": 2,
                    "score": current_score,
                }
            )

        self.discard_pile = [self.deck.pop()]
        self.current_player = 0
        self.done = False
        self.final_player = None
        self.pending_card = None
        self.previous_action = ""
        return self._get_obs()

    def _get_obs(self) -> dict:
        """
        Returns the current observation of the environment.

        Returns:
            dict: A dictionary with:
                - players (list): list of player states (grid, visible, etc.)
                - discard (list): discard pile
                - deck_count (int): number of cards left in the deck
                - current_player (int): index of the player to play
        """
        return {
            "players": self.players,
            "discard": self.discard_pile,
            "deck_count": len(self.deck),
            "current_player": self.current_player,
            "pending_card": self.pending_card,
            "previous_action": self.previous_action,
        }

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Applies one step in the environment. This function operates in two phases:
        - Phase 1: Drawing a card (action={"draw": "deck"/"discard"})
        - Phase 2: Using that card (action={"replace": int or "flip": int}

        If the deck is empty, the discard pile is shuffled and becomes the new deck.

        Args:
            action (dict): Either:
                - {"draw": "deck" or "discard"} to draw a card
                - {"replace": int (0-11) or "flip": int (0-11)} to use the drawn card

        Returns:
            tuple: (observation, reward, done, info)
        """
        if "flip" in action and self.previous_action == "draw_discard":
            raise ValueError(
                "You cannot flip a card after drawing from the discard pile. You must replace."
            )

        player = self.players[self.current_player]

        # PHASE 1: Drawing the card
        if "draw" in action:
            if self.deck == []:
                self.deck = self.discard_pile
                random.shuffle(self.deck)
                self.discard_pile = [self.deck.pop()]

            if self.pending_card is not None:
                raise ValueError("You must use the previously drawn card first.")
            if action["draw"] == "deck":
                self.pending_card = self.deck.pop()
                self.previous_action = "draw_deck"
            elif action["draw"] == "discard":
                self.pending_card = self.discard_pile.pop()
                self.previous_action = "draw_discard"
            else:
                raise ValueError("Invalid draw source.")
            return self._get_obs(), 0.0, self.done, {}

        # PHASE 2: Using the drawn card
        if self.pending_card is None:
            raise ValueError("No pending card. Please draw first.")

        if "replace" not in action and "flip" not in action:
            raise ValueError("Missing 'replace' or 'flip' in action.")
        if "replace" in action and "flip" in action:
            raise ValueError("Cannot provide both 'replace' and 'flip' in action.")

        replace_index = action.get("replace", None)
        flip_index = action.get("flip", None)

        if replace_index is not None:
            # Replace a card
            if player["grid"][replace_index] is None:
                raise ValueError(
                    f"Invalid replace: position {replace_index} is already removed."
                )
            self.discard_pile.append(player["grid"][replace_index])
            player["score"] -= player["grid"][replace_index]
            player["score"] += self.pending_card
            if not player["visible"][replace_index]:
                player["return_count"] += 1
            player["grid"][replace_index] = self.pending_card
            player["visible"][replace_index] = True

        else:
            # Discard the drawn card, and flip the selected hidden card
            if player["grid"][flip_index] is None:
                raise ValueError(f"Cannot flip index {flip_index}: already removed.")
            if player["visible"][flip_index]:
                raise ValueError(f"Cannot flip index {flip_index}: already visible.")
            player["score"] += player["grid"][flip_index]
            player["visible"][flip_index] = True
            player["return_count"] += 1
            self.discard_pile.append(self.pending_card)

        # Reset pending card
        self.pending_card = None

        # Column removal
        self._check_and_remove_columns(player)

        # Final round logic
        if player["return_count"] == CARD_COUNT and self.final_player is None:
            self.final_player = self.current_player
            self.remaining_final_turns = self.num_players

        # Next player
        self.current_player = (self.current_player + 1) % self.num_players
        if self.final_player is not None:
            self.remaining_final_turns -= 1
            if self.remaining_final_turns == 0:
                self.done = True

        # Remove last action
        self.previous_action = ""

        return self._get_obs(), 0.0, self.done, {}

    def _check_and_remove_columns(self, player: dict) -> None:
        """
        Checks the player's grid for any column where all 3 visible cards
        are the same and removes that column (sets values to None).

        Args:
            player (dict): The player whose grid is checked and potentially modified.
        """
        grid = np.array(player["grid"]).reshape(3, 4)
        visible = np.array(player["visible"]).reshape(3, 4)
        for col in range(4):
            col_vals = grid[:, col]
            col_vis = visible[:, col]
            if all(col_vis) and len(set(col_vals)) == 1:
                # On retire les cartes de cette colonne
                for row in range(3):
                    idx = row * 4 + col
                    self.discard_pile.append(player["grid"][idx])
                    player["grid"][idx] = None  # plus de carte
                    player["visible"][idx] = False

    def render(self):
        """
        Displays the current state of the game in a human-readable format.
        Shows each player's grid, the top discard card, and whose turn it is.
        """
        print("=== SKYJO ===")
        for i, p in enumerate(self.players):
            line = f"Joueur {i}: "
            for val, vis in zip(p["grid"], p["visible"]):
                v = "--" if val is None else (val if vis else "??")
                line += f"{str(v).rjust(4)}"
            print(line)
        print(f"Défausse: {self.discard_pile[-1]}")
        if self.final_player is not None:
            print(
                f"Fin déclenchée par joueur {self.final_player} - {self.remaining_final_turns} tours restants"
            )
        print(f"À jouer: Joueur {self.current_player}")
        print()

    def get_final_scores(self) -> dict[int, int]:
        """
        Calculate the final scores for each player when the game is done.
        
        Rules:
        - For each player, the score is the sum of their remaining cards
        - If the player who ended the game doesn't have the lowest score:
            - Their score is doubled (if positive)
            - Their score remains unchanged if negative
        - If multiple players have the same lowest score, the player who ended the game
          gets their score doubled
        
        Returns:
            dict[int, int]: A dictionary mapping player indices to their final scores
        """
        if not self.done:
            raise ValueError("Cannot calculate final scores before the game is done")
            
        # Calculate raw scores (sum of remaining cards)
        final_scores = {}
        for i, player in enumerate(self.players):
            score = sum(card for card in player["grid"] if card is not None)
            final_scores[i] = score
            
        # Find the lowest score
        lowest_score = min(final_scores.values())
        
        # Apply penalty to the player who ended the game if they don't have the lowest score
        if self.final_player is not None:
            if final_scores[self.final_player] > lowest_score:
                final_scores[self.final_player] *= 2
            else:
                for player_id, score in final_scores.items():
                    if score == lowest_score and player_id != self.final_player:
                        final_scores[self.final_player] *= 2
            
        else:
            raise ValueError("No final player found. The game is not done.")
                    
        return final_scores
