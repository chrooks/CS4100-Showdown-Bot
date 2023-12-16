"""
This module defines the 'Dojo' class, a custom environment player for Pokémon Showdown 
battles, specifically tailored for Generation 8 games. It extends the Gen8EnvSinglePlayer 
class from the poke-env library and includes methods for calculating rewards, embedding 
battle states into numerical vectors, and defining the observation space for reinforcement 
learning models. The class aims to provide a comprehensive and detailed representation of 
the battle state, including individual Pokémon features, side conditions, weather, and 
field effects, to facilitate effective training of a reinforcement learning agent.
"""
from typing import Dict
import numpy as np
from gym.spaces import Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.field import Field
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move
from poke_env.environment.side_condition import SideCondition, STACKABLE_CONDITIONS
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.data.gen_data import GenData


class Dojo(Gen8EnvSinglePlayer):
    """
    Customized environment player for Pokémon Showdown battles, built for
    Generation 8 games. It extends Gen8EnvSinglePlayer and includes methods
    for reward calculation, state embedding, and defining the observation space.
    """

    def calc_reward(self, prev_turn: AbstractBattle, current_turn: AbstractBattle):
        reward = 0

        # Won game
        if current_turn.won:
            reward += 300

        # Lost game
        if current_turn.lost:
            reward -= 300

        # Fainted opposing Pokemon
        prev_fainted_opponents = sum(
            1 for mon in prev_turn.opponent_team.values() if mon.fainted)
        current_fainted_opponents = sum(
            1 for mon in current_turn.opponent_team.values() if mon.fainted)
        if current_fainted_opponents > prev_fainted_opponents:
            # Adjust reward magnitude as needed
            reward += (current_fainted_opponents - prev_fainted_opponents) * 80

        # Allowed pokemon to be Fainted
        prev_fainted_self = sum(
            1 for mon in prev_turn.team.values() if mon.fainted)
        current_fainted_self = sum(
            1 for mon in current_turn.team.values() if mon.fainted)
        if current_fainted_self > prev_fainted_self:
            reward -= (current_fainted_self - prev_fainted_self) * \
                60  # Adjust penalty magnitude as needed

        # Doing a lot of damage / Taking a lot of damage
        opponent_damage_taken = 0
        self_damage_taken = 0

        for mon_name, mon in current_turn.opponent_team.items():
            if mon_name in prev_turn.opponent_team:
                opponent_damage_taken += prev_turn.opponent_team[mon_name].current_hp - mon.current_hp

        for mon_name, mon in current_turn.team.items():
            if mon_name in prev_turn.team:
                self_damage_taken += prev_turn.team[mon_name].current_hp - mon.current_hp

        # Adjust these values as necessary
        reward += opponent_damage_taken * 0.1  # Reward for doing damage
        reward -= self_damage_taken * 0.1      # Penalty for taking damage

        # Switched to a Pokémon with a better defensive type matchup
        prev_active_pokemon = prev_turn.active_pokemon
        current_active_pokemon = current_turn.active_pokemon
        opposing_pokemon = current_turn.opponent_active_pokemon

        prev_active_pokemon = prev_turn.active_pokemon
        current_active_pokemon = current_turn.active_pokemon
        opposing_pokemon = current_turn.opponent_active_pokemon

        if prev_active_pokemon.species != current_active_pokemon.species:
            # Check if the new Pokémon's type resists or is weak against the opposing Pokémon's type
            for move in opposing_pokemon.moves.values():
                damage_multiplier = self.calc_damage_multiplier(move, current_active_pokemon)
                prev_damage_multiplier = self.calc_damage_multiplier(move, prev_active_pokemon)

                if damage_multiplier < 1:
                    reward += 30  # Reward for resisting the opponent's move type
                elif damage_multiplier > 1:
                    reward -= 15  # Penalty for weakness to the opponent's move type

                # Additional bonus if new Pokémon resists a weakness of the previous Pokémon
                if prev_damage_multiplier > 1 and damage_multiplier < 1:
                    reward += 20  # Additional bonus for a strategic switch
                elif prev_damage_multiplier < 1 and damage_multiplier > 1:
                    reward -= 10  # Penalty for switching into a disadvantageous matchup

                # Check for higher or lower defense or special defense
                if opposing_pokemon.base_stats['atk'] > opposing_pokemon.base_stats['spa']:
                    # Opposing Pokémon has a higher physical attack, compare defense
                    if current_active_pokemon.stats['def'] > prev_active_pokemon.stats['def']:
                        reward += 15  # Reward for switching to a Pokémon with higher defense
                    elif current_active_pokemon.stats['def'] < prev_active_pokemon.stats['def']:
                        reward -= 10  # Penalty for switching to a Pokémon with lower defense
                else:
                    # Opposing Pokémon has a higher special attack, compare special defense
                    if current_active_pokemon.stats['spd'] > prev_active_pokemon.stats['spd']:
                        reward += 15  # Reward for switching to a Pokémon with higher special defense
                    elif current_active_pokemon.stats['spd'] < prev_active_pokemon.stats['spd']:
                        reward -= 10  # Penalty for switching to a Pokémon with lower special defense
        return reward

    def calc_damage_multiplier(self, move: Move, opponent: Pokemon):
        # Calculating the damage multiplier
        return move.type.damage_multiplier(
            opponent.type_1,
            opponent.type_2,
            type_chart=GenData.from_gen(8).type_chart,
        )

    def embed_battle(self, battle: AbstractBattle):
        """
        Embeds the current battle state into a numerical vector suitable for machine learning models
        The vector includes features of each Pokémon in both the player's and the opponent's team,
        along with side conditions, weather, and field effects.

        Args:
            battle (AbstractBattle): The current battle state.

        Returns:
            numpy.array: The state vector representing the current battle state.
        """
        state_vector = []

        max_team_size = 6  # Maximum number of Pokémon per team
        pokemon_feature_length = 70  # Length of features for each Pokémon

        # Add attributes for each Pokémon in your team
        for mon in battle.team.values():
            state_vector.extend(
                self._extract_pokemon_features(mon, battle)
            )  # 70 elements each

        # Pad your team's features if team size is less than max_team_size
        for _ in range(max_team_size - len(battle.team)):
            # Padding with zeros
            state_vector.extend([0] * pokemon_feature_length)

        # Add attributes for each Pokémon in the opponent's team
        for mon in battle.opponent_team.values():
            state_vector.extend(
                self._extract_pokemon_features(mon, battle, opponent=True)
            )  # 70 elements each

        # Pad opponent's team features if team size is less than max_team_size
        for _ in range(max_team_size - len(battle.opponent_team)):
            # Padding with zeros
            state_vector.extend([0] * pokemon_feature_length)

        # Add attributes for each player's side
        side_features = self._extract_side_features(battle)
        # Add battlefield attributes
        weather_features = self._encode_weather(battle.weather)
        fields_features = self._encode_fields(battle.fields)

        state_vector.extend(side_features)  # 20 elements [0-3]
        state_vector.extend(weather_features)  # 9 elements [0-1]
        state_vector.extend(fields_features)  # 13 elements [0-1]

        return np.array(state_vector, dtype=np.float32)

    def describe_embedding(self):
        """
        Defines the structure and limits of the observation space for the reinforcement
        learning environment. The observation space is represented by a Box space,
        with low and high bounds for each feature in the state vector.

        Returns:
            gym.spaces.Box: The Box space representing the observation space.
        """
        num_pokemon = 12  # Number of Pokémon considered
        # 20 (Players' side) + 9 (Weather) + 13 (Field)
        additional_features = 42

        # Constructing the low vector
        pokemon_low_vector = (
            [0.0, 0.0]
            + [0] * 36
            + [0] * 7
            + [-1] * 7
            + [0.0] * 6
            + [-1] * 4
            + [-1] * 4
            + [-1] * 4
        )
        low = pokemon_low_vector * num_pokemon + [0] * additional_features

        # Constructing the high vector
        pokemon_high_vector = (
            [1.0, 1.0]
            + [1] * 36
            + [1] * 7
            + [1] * 7
            + [1.0] * 6
            + [1.0] * 4
            + [1.0] * 4
            + [4.0] * 4
        )
        high = pokemon_high_vector * num_pokemon + \
            [3] * 20 + [1] * 9 + [1] * 13

        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _extract_pokemon_features(
        self, mon: Pokemon, battle: AbstractBattle, opponent=False
    ):
        """
        Extracts features of a given Pokémon in a battle. Features include current HP,
        level, type, status conditions, stat boosts, move effectiveness, base power, and accuracy.
        If the Pokémon is an opponent, move data is excluded.

        Args:
            mon (Pokemon): The Pokémon for which to extract features.
            battle (AbstractBattle): The current battle state.
            opponent (bool): Flag indicating if the Pokémon is an opponent.

        Returns:
            List[float]: The list of extracted features.
        """
        # Extracting basic features
        current_hp = mon.current_hp_fraction  # Current HP as a fraction of total HP
        level = mon.level / 100  # Normalize level

        # Pokemon types one-hot encoding
        num_types = 18  # Total number of Pokémon types
        # Initialize a zero vector for two types
        types_vector = [0] * num_types * 2
        if mon.type_1:
            # Set 1 at the index corresponding to type_1
            types_vector[mon.type_1.value - 1] = 1
        if mon.type_2:
            # Set 1 at the index corresponding to type_2
            types_vector[num_types + mon.type_2.value - 1] = 1

        # Extracting status condition
        # Initialize a list for all possible status conditions
        status_condition = [0] * len(Status)
        if mon.status:
            # Set the corresponding status to 1
            status_condition[mon.status.value - 1] = 1

        # Extracting and normalizing stats
        stats = [
            mon.base_stats[stat] / 255  # Normalize boosts
            for stat in ["hp", "atk", "def", "spa", "spd", "spe"]
        ]

        # Extracting stat boosts
        boosts = [
            mon.boosts[stat] / 6  # Normalize boosts
            for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        ]

        # Initializing move data for all cases
        moves_base_power = [-1] * 4  # Default value for unavailable moves
        moves_accuracy = [-1] * 4  # Default value for unavailable moves
        moves_effectiveness = [-1] * 4  # Default value for move effectiveness

        if not opponent:
            for i, move in enumerate(mon.moves.values()):
                # Normalizing base power and accuracy, and ensuring index bounds
                if i < 4:
                    moves_base_power[i] = (
                        move.base_power / 250 if move.base_power else -1
                    )
                    moves_accuracy[i] = move.accuracy if move.accuracy else -1

                    # Calculating effectiveness if move type is known
                    if move.type:
                        effectiveness = self.calc_damage_multiplier(
                            move, battle.opponent_active_pokemon)
                        moves_effectiveness[i] = effectiveness

        # Combining all features into a single state vector
        state_vector = [
            current_hp,  # 1 element [0.0, 1.0]
            level,  # 1 element [0.0, 1.0]
            *types_vector,  # 36 elements (18 * 2) [0, 1]
            *status_condition,  # 7 elements [0, 1]
            *boosts,  # 7 elements [-1, 1]
            *stats,  # 6 elements [0.0, 1]
            *moves_base_power,  # 4 elements [-1, 1.0]
            *moves_accuracy,  # 4 elements [-1, 1.0]
            *moves_effectiveness,  # 4 elements [-1, 4.0]
        ]

        return state_vector

    def _extract_side_features(self, battle: AbstractBattle):
        """
        Extracts side condition features from both the player's and opponent's sides.
        Features include various in-battle conditions such as spikes, stealth rock, etc.

        Args:
            battle (AbstractBattle): The current battle state.

        Returns:
            List[int]: The list of extracted side condition features.
        """
        # Initialize all side conditions with 0
        side_features = {condition: 0 for condition in SideCondition}

        # Update based on actual side conditions in the battle
        for condition, value in battle.side_conditions.items():
            side_features[condition] = min(
                value, STACKABLE_CONDITIONS.get(condition, 1)
            )

        for condition, value in battle.opponent_side_conditions.items():
            side_features[condition] = min(
                value, STACKABLE_CONDITIONS.get(condition, 1)
            )

        # Convert to list
        return [side_features[condition] for condition in SideCondition]

    def _encode_weather(self, weather: Dict[Weather, int]):
        """
        Encodes the current weather condition in the battle into a binary vector.

        Args:
            weather (Dict[Weather, int]): A dictionary mapping the current weather
                                           condition to its starting turn.

        Returns:
            List[int]: The binary vector representing the weather condition.
        """
        weather_vector = [0] * len(Weather)  # Initialize vector with zeros
        for active_weather in weather.keys():
            # Set the corresponding index to 1
            weather_vector[active_weather.value - 1] = 1
        return weather_vector

    def _encode_fields(self, fields: Dict[Field, int]):
        """
        Encodes the current field effects in the battle into a binary vector.

        Args:
            fields (Dict[Field, int]): A dictionary mapping the current field effects
                                       to their activation turns.

        Returns:
            List[int]: The binary vector representing the field effects.
        """
        fields_vector = [0] * len(Field)  # Initialize vector with zeros
        for active_field in fields.keys():
            # Set the corresponding index to 1
            fields_vector[active_field.value - 1] = 1
        return fields_vector
