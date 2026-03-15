"""
Dataset Generator — produces 500 synthetic user profiles.
Column names are consistent across all pipeline modules.
"""
import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

DOMAINS = ["Computer Science","Mathematics","Psychology","Engineering",
           "Biology","Business","Literature","Physics","Data Science","Education"]

SKILLS_POOL = ["Python","Statistics","Writing","Research","Communication",
               "Problem Solving","Data Analysis","Leadership","Critical Thinking",
               "Machine Learning","Project Management","Presentation",
               "Programming","Teamwork","Time Management"]

AVAILABILITY_SLOTS = ["Morning","Afternoon","Evening","Night","Weekends","Flexible"]
EXPERIENCE_LEVELS  = ["Beginner","Intermediate","Advanced","Expert"]

EMOTIONAL_TEXTS = [
    "I feel really overwhelmed lately and don't know how to cope.",
    "Everything seems too much. I feel isolated and exhausted.",
    "I've been struggling with anxiety and can't focus on anything.",
    "I feel hopeless about my progress and don't see a way forward.",
    "I'm really stressed and feeling disconnected from everyone.",
    "I don't feel motivated at all and keep withdrawing from activities.",
    "Today was hard. I felt very lonely and couldn't concentrate.",
    "I'm feeling burnt out and emotionally drained.",
]
ACADEMIC_TEXTS = [
    "I'm struggling to understand the core concepts in my coursework.",
    "I failed my last assignment and don't know where to improve.",
    "I need help organizing my study plan and understanding the material.",
    "My grades have been dropping and I feel academically behind.",
    "I can't grasp the mathematical concepts no matter how hard I try.",
    "I need a study partner who can explain things differently.",
    "I'm falling behind on deadlines and need academic guidance.",
    "I don't understand how to approach the research methodology.",
]
MOTIVATIONAL_TEXTS = [
    "I know what I need to do but I just can't get myself started.",
    "I lack the energy to begin my tasks even though I understand them.",
    "I keep procrastinating and need someone to hold me accountable.",
    "I want to improve but can't find the motivation to push forward.",
    "I need someone to inspire me and keep me on track.",
    "I feel neutral — not bad, just stuck and unmotivated.",
    "I need a push to stay consistent with my goals.",
    "I have the knowledge but struggle with execution and drive.",
]
NEUTRAL_TEXTS = [
    "Things are going okay. Just checking in today.",
    "Had a normal day. Nothing special to report.",
    "Feeling alright. Working through my tasks steadily.",
    "Today was balanced. Making good progress overall.",
    "Everything is fine. Learning and growing steadily.",
    "Doing well. Staying on track with my goals.",
    "Pretty good day. Focused and productive.",
    "Things are manageable. Looking forward to improving more.",
]


def generate_dataset(n: int = 500, save_path: str = None) -> pd.DataFrame:
    archetypes = ["emotional","academic","motivational","neutral"]
    records = []
    for i in range(n):
        arch = random.choice(archetypes)
        domain   = random.choice(DOMAINS)
        skills   = ", ".join(random.sample(SKILLS_POOL, random.randint(2, 5)))
        avail    = ", ".join(random.sample(AVAILABILITY_SLOTS, random.randint(1, 3)))
        exp      = random.choice(EXPERIENCE_LEVELS)

        if arch == "emotional":
            mood    = round(random.uniform(1.0, 4.0), 1)
            eng     = round(random.uniform(0.1, 0.45), 2)
            delta   = round(random.uniform(-0.3, -0.05), 2)
            text    = random.choice(EMOTIONAL_TEXTS)
            true_sp = "Emotional"
        elif arch == "academic":
            mood    = round(random.uniform(3.5, 6.5), 1)
            eng     = round(random.uniform(0.15, 0.5), 2)
            delta   = round(random.uniform(-0.25, -0.05), 2)
            text    = random.choice(ACADEMIC_TEXTS)
            true_sp = "Academic"
        elif arch == "motivational":
            mood    = round(random.uniform(4.0, 6.5), 1)
            eng     = round(random.uniform(0.25, 0.60), 2)
            delta   = round(random.uniform(-0.15, 0.05), 2)
            text    = random.choice(MOTIVATIONAL_TEXTS)
            true_sp = "Motivational"
        else:
            mood    = round(random.uniform(6.0, 10.0), 1)
            eng     = round(random.uniform(0.6, 1.0), 2)
            delta   = round(random.uniform(0.0, 0.2), 2)
            text    = random.choice(NEUTRAL_TEXTS)
            true_sp = "None"

        records.append({
            "UserID":          f"User_{i+1:04d}",
            "Mood":            mood,
            "Engagement":      eng,
            "EngagementDelta": delta,
            "Reflection":      text,
            "Domain":          domain,
            "Skills":          skills,
            "Experience":      exp,
            "Availability":    avail,
            "TrueSupport":     true_sp,
        })

    df = pd.DataFrame(records)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved: {save_path}")
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peer_support_dataset.csv")
    df = generate_dataset(n=500, save_path=out)
    print(f"Total users  : {len(df)}")
    print(df["TrueSupport"].value_counts().to_string())
