import gym
from gym import spaces
from transformers import GPT2Tokenizer
import numpy as np

class ChatbotEnv(gym.Env):
    """Custom environment for the chatbot.

    This environment simulates a conversation between a user and a chatbot.
    The chatbot receives an action (a token) from the user and generates a response.
    The environment provides observations, rewards, and tracks the conversation history.

    Parameters
    ----------
        tokenizer : GPT2Tokenizer
            The tokenizer used to encode and decode text.
        action_space : gym.spaces.Discrete
            The action space representing the possible tokens.
        observation_space : gym.spaces.Box
            The observation space representing the conversation history.
        history : list
            The list of messages exchanged between the user and the chatbot.
    """

    def __init__(self):
        super(ChatbotEnv, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # add a padding token if the tokenizer does not have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)
        self.observation_space = spaces.Box(low=0, high=self.tokenizer.vocab_size, shape=(1024,), dtype=np.int32)
        self.history = []

    def reset(self):
        """Resets the environment and returns the initial observation.

        Returns
        -------

            np.ndarray:
                The initial observation representing the conversation history.
        """
        self.history = ["initial message"]
        initial_observation = self._get_observation()
        return initial_observation

    def step(self, action):
        """Takes a step in the environment based on the given action.

        Parameters
        ----------
            action : int
                The action (token) chosen by the user.

        Returns
        -------

        tuple:
            A tuple containing the next observation, reward, done flag, and additional information.
            - observation (np.ndarray): The next observation representing the updated conversation history.
            - reward (float): The reward obtained from the chatbot's response.
            - done (bool): A flag indicating if the conversation is finished.
            - info (dict): Additional information about the step.
        """
        response = self.tokenizer.decode(action, skip_special_tokens=True)
        self.history.append(response)
        observation = self._get_observation()
        reward = self._get_reward(response)
        done = len(self.history) > 20
        return observation, reward, done, {}

    def _get_observation(self):
        """Flattens the conversation history into a single input tensor.

        Returns
        -------
            np.ndarray: The flattened input tensor representing the conversation history.
        """
        flat_history = self.tokenizer.encode(" ".join(self.history), max_length=1024, padding='max_length', truncation=True, return_tensors='np')
        return flat_history[0]

    def _get_reward(self, response):
        """Calculates the reward for the chatbot's response.

        The reward is based on the length of the response, the presence of specific keywords,
        and the informativeness of the response.

        Parameters
        ----------

            response (str): The chatbot's response.

        Returns
        -------
        
            float: The reward obtained from the chatbot's response.
        """
        length = len(response.split())
        if length < 5:
            length_reward = -1
        elif length > 20:
            length_reward = -1
        else:
            length_reward = 1

        # let's reward our model being polite ! :D
        keywords = ['thank', 'please', 'sorry']
        keyword_reward = sum(1 for word in keywords if word in response.lower())

        # add penalty for uninformative responses
        uninformative_responses = ['i don\'t know', 'idk', 'no comment']
        uninformative_penalty = -1 if any(phrase in response.lower() for phrase in uninformative_responses) else 0

        # reward calculation for the response
        total_reward = length_reward + keyword_reward + uninformative_penalty
        return total_reward


