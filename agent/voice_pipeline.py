import asyncio
from .stt import transcribe
from .llm import ConversationAgent
from .tts import synthesize

class VoicePipeline:
    def __init__(self, voice='af_heart', speed=1.0):
        self.agent = ConversationAgent()
        self.current_voice = voice
        self.speed = speed
        self.is_running = False
        
    async def process_turn(self, user_audio_bytes: bytes, on_transcript, on_response, on_audio):
        self.is_running = True
        try:
            # 1. Transcribe (run in thread to not block event loop)
            transcript_text = await asyncio.to_thread(transcribe, user_audio_bytes)
            if not transcript_text:
                return
                
            # Await callback if async, else call
            if asyncio.iscoroutinefunction(on_transcript):
                await on_transcript(transcript_text)
            else:
                on_transcript(transcript_text)
                
            # 2. Get LLM response
            response_text = await asyncio.to_thread(self.agent.get_response, transcript_text)
            
            if asyncio.iscoroutinefunction(on_response):
                await on_response(response_text)
            else:
                on_response(response_text)
                
            # 3. Synthesize Audio
            audio_bytes = await asyncio.to_thread(synthesize, response_text, self.current_voice, self.speed)
            
            if asyncio.iscoroutinefunction(on_audio):
                await on_audio(audio_bytes)
            else:
                on_audio(audio_bytes)
                
        finally:
            self.is_running = False

    def set_voice(self, voice_id: str):
        self.current_voice = voice_id
        
    def reset(self):
        self.agent.reset()
