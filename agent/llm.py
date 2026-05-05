import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

SYSTEM_PROMPT = """You are a creative workshop facilitator for Productive Play, a company that uses LEGO-based business simulations for team-building workshops. Your role is to help teams brainstorm, solve problems, collaborate, and think creatively. You guide discussions using interactive facilitation techniques. IMPORTANT: Your responses will be spoken aloud by a voice agent, so keep every response to 2-3 sentences maximum. Be warm, encouraging, and concise."""

class ConversationAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.history = []
        
    def get_response(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=self.history
        )
        
        assistant_text = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_text})
        
        return assistant_text
        
    def reset(self):
        self.history = []

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_key_here":
        print("Please set ANTHROPIC_API_KEY in .env to test the LLM module.")
    else:
        agent = ConversationAgent()
        print("Testing LLM Agent...")
        user_msg = "Hello! I'm ready for our team building workshop."
        print(f"User: {user_msg}")
        response = agent.get_response(user_msg)
        print(f"Agent: {response}")
