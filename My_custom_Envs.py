import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import nltk


class HangedManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "n_total_try_games": 11}

    """Gym environment for the Hanged Man game.
    Args:
        max_word_size (int): maximum size of the word to be guessed
        word_dictionary (set): Dictionary of words
    Attributes:
        max_word_size (int): maximum size of the word to be guessed
        n_total_try_games (int): Maximum number of try
        word_dictionary (set): Dictionary of words
        encoded_state (np.array int): Current state encoded.
        encoded_tried_letters (np.array bool):
        encoded_aim (np.array int): Current word to guess encoded. When the encoded
        state reach this aim, the game ends.
    """

    def __init__(self, max_word_size=8, word_dictionary=None, render_mode=None):

        # Specifics attributes
        self.n_total_try_games = self.metadata["n_total_try_games"]
        self.encoded_state = np.empty(0)
        self.encoded_aim = np.empty(0)
        self.encoded_tried_letters = None
        self._nb_left_try = None  # number of try left
        self.word_dictionary = word_dictionary
        self.max_word_size = max_word_size

        # Action Space
        # The agent can choose between all letter.
        self.action_space = spaces.Discrete(26)

        # Observations Space
        
        # 26 for each letter of the alphabet.
        # 1 for "to be guessed" (represented as _).
        # 0 for "not active in this game" (represented as a padding).
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([28] * self.max_word_size),  # Encoded word   # 28 states for each position in the word
            spaces.MultiBinary(26)  # Binary status for letters
        ))

        if self.word_dictionary is None:
            # By default, we use nltk dictionary 
            try: 
                nltk.data.find('corpora/words.zip') 
            except LookupError:
                nltk.download('words')
            word_list = nltk.corpus.words.words()  # this returns a list
            self.word_dictionary = [word for word in word_list if len(word) <= max_word_size]

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.window = None
        self.clock = None

    def encode_word(self, word):
        """Convert word string to encoded numpy array."""
        encoded = np.zeros(self.max_word_size, dtype=np.int8)

        for idx, char in enumerate(word):
            if char == '_':
                encoded[idx] = 1
            else:
                encoded[idx] = ord(char) - ord('A') + 2
        return encoded

    @staticmethod
    def decode_word(encoded_word):
        """Convert numpy array back to word string."""
        decoded = ""

        for code in encoded_word:
            if code == 0:
                break
            elif code == 1:
                decoded = decoded + "_"
            else:
                decoded = decoded + chr(code + ord('A') - 2)
        return decoded

    def decode_tried_letters(self):
        """"Decode encoded letters for human rendering"""
        tried_letters = set()

        for index, tried in enumerate(self.encoded_tried_letters):
            if tried:
                letter = chr(index + ord('A'))
                tried_letters.add(letter)

        return tried_letters

    def _get_obs(self):
        """" Get observation """
        return (self.encoded_state, self.encoded_tried_letters)
 

    def _get_info(self):
        """" Get info """
        return {'left try': self._nb_left_try}  

    def reset(self, seed=None, **kwargs):
        """
        Set or reset the environment attributes to start a new episode
        :param seed: To fixe a seed
        """
        super().reset(seed=seed)

        # Observations Space
        # Set encodes states with all elements inactive (=0)
        self.encoded_state = np.zeros(self.max_word_size, dtype=np.uint8)

        # Sample a word from dict
        word = random.choice(self.word_dictionary )
        while len(word) > self.max_word_size:
            word = random.choice(self.word_dictionary)
        decoded_word = word.upper()

        # Set up active elements in encoded state
        for i in range(0, len(decoded_word)):
            self.encoded_state[i] = 1
        
        # Encode the word that will be the aim for the encoded state
        self.encoded_aim = self.encode_word(decoded_word)
        # a bool array to know which letters have been tried
        self.encoded_tried_letters = np.zeros(26, dtype=np.bool_)
        # Set the nb_left_try
        self._nb_left_try = self.n_total_try_games

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        """" Process the game, maximize the encoded stuff for performance"""
        # Check action correct type
        try: 
            action = int(action)
        except :
            print("Action not valid, must be an int : 0 for A, 1 for B, ... ")
            return self._get_obs(), 0, False, False, self._get_info()
        # Check if letter already tried an within the rane 
        if action >=26 or action < 0 :
            print("letter out of bound : : 0 for A, 1 for B, ...")
            return self._get_obs(), 0, False, False, self._get_info()
        if self.encoded_tried_letters[action] :
            print("letter already tried")
            return self._get_obs(), 0, False, False, self._get_info()

        # negative reward because the aim to win the fastest
        # if a letter is found then the reward will be 0 for the step
        reward = -1
        

        # Update la liste of tried letters
        self.encoded_tried_letters[action] = True
        encoded_letter = action + 2

        # Find the indices in encoded_aim where the letter is present
        letter_indices = np.where(self.encoded_aim == encoded_letter)[0]
        if len(letter_indices) > 0:  # If indices are found (letter exists in encoded_aim)
            self.encoded_state[letter_indices] = encoded_letter   # Update the encoded_state at these indices
            reward = 0          # Set reward to zero if letter found
        
        # Update
        self._nb_left_try += reward

        # Terminated ?
        terminated = self._nb_left_try == 0 or np.array_equal(self.encoded_state, self.encoded_aim)


        truncated = False  # often time limit, not use in this contexte

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            stages = [
         """
 
               
           
           
        
           
        --------
        """,
         """
 
           |    
           |
           |
           |
           |
        --------
        """,
        """
           ------
           |    
           |
           |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |
           |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |    |
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|\\
           |
           |
        --------
        """,
        """
           ------
           |    |
           |    O
           |   /|\\
           |   /
           |
        --------
        """,
            """
            ------
            |    |
            |    O
            |   /|\\
            |   / \\
            |
            --------
            """
            ]
        n_errors = self.n_total_try_games - self._nb_left_try  
        if n_errors >= len(stages):
            n_errors = len(stages) - 1 

        # Affichage du dessin correspondant au nombre d'erreurs
        print(stages[n_errors])

        # Affichage du mot deviné et des lettres essayées
        print("Guessed word  : " + self.decode_word(self.encoded_state))
        print("Tried letters : " + str(self.decode_tried_letters()))
        # For other render modes, like returning an array for a graphical interface