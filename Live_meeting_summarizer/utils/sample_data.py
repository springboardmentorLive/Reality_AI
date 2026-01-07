DEFAULT_MEETING_TRANSCRIPT = """
Alice (Product Manager): "Alright team, let's get started. The goal of this meeting is to finalize the launch plan for the new 'Live Summarizer' feature. Bob, how is the backend looking?"

Bob (Lead Developer): "We're mostly on track. The API endpoints for the audio upload are capable of handling concurrent requests now. I did run into a small issue with the latency on the transcription service, though. It's averaging about 5 seconds for a 30-second clip."

Alice: "5 seconds is a bit high. Can we optimize that?"

Bob: "Yes, I think we can cache the model in memory. I'll add that to my task list for tomorrow. It should cut the time in half."

Charlie (UX Designer): "From the design side, the new 'Dark Mode' UI is ready. I've handed off the assets to the frontend team. One question: are we sticking with the blue accent color or switching to the new teal?"

Alice: "Let's stick with the blue for MVP to maintain consistency. We can A/B test the teal later."

Charlie: "Got it. I'll update the style guide to reflect that decision."

Alice: "Okay, moving on to marketing. Sarah isn't here, but she sent an update saying the blog post draft is ready for review. I need everyone to look at it by Thursday."

Bob: "Will do. Also, are we still aiming for a soft launch next Monday?"

Alice: "Yes, that's the plan. Soft launch on Monday to internal users, then public beta two weeks later. Bob, please ensure the staging environment is stable by Friday for the internal demo."

Bob: "Understood. I'll freeze the code on Thursday night."

Alice: "Perfect. To recap: Bob is optimizing the API latency and ensuring staging is ready. Charlie is finalizing the style guide with the blue accent. Everyone needs to review Sarah's blog post. Thanks everyone, let's break."
"""
