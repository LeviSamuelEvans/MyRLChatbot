# tests/test_environment.py
import unittest
from src.environment import ChatbotEnv

class TestChatbotEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChatbotEnv()

    def test_reset(self):
        initial_state = self.env.reset()
        self.assertEqual(len(initial_state), 0)

    def test_step(self):
        self.env.reset()
        action = self.env.tokenizer.encode("Hello", return_tensors='pt').numpy()[0]
        _, reward, _, _ = self.env.step(action)
        self.assertTrue(reward > 0)

if __name__ == "__main__":
    unittest.main()
