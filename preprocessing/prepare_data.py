import pandas as pd
from typing import List, Dict
import json
import re

def load_conversations(csv_path: str) -> pd.DataFrame:
    """Load the conversation CSV file"""
    df = pd.read_csv(csv_path)
    if len(df.columns) != 1:
        raise ValueError("Expected CSV file to have exactly one column")
    return df

def split_message(message: str) -> tuple:
    """Split a message into speaker and content
    Example: "121415, 33956 AM Dad🤗💼 message" -> ("dad", "message")
    """
    # Match timestamp, speaker name, and message
    # This regex matches:
    # - Numbers and commas for timestamp
    # - AM/PM
    # - Either "Dad🤗💼" or "Harold"
    # - The rest is the message
    pattern = r'[\d,\s]+(?:AM|PM)\s+(Dad🤗💼|Harold)\s+(.+)$'
    match = re.match(pattern, str(message).strip())
    
    if not match:
        return None, None
    
    speaker = match.group(1).strip()
    # Convert "Dad🤗💼" to just "dad" for consistency
    speaker = 'dad' if '🤗' in speaker else speaker.lower()
    content = match.group(2).strip()
    return speaker, content

def create_training_pairs(df: pd.DataFrame) -> List[Dict]:
    """Convert conversations into training pairs"""
    training_pairs = []
    messages = []
    
    message_column = df.columns[0]
    
    for i in range(len(df)):
        raw_message = df.iloc[i][message_column]
        speaker, content = split_message(raw_message)
        
        # Skip invalid messages
        if speaker is None or not content:
            print(f"Skipping invalid message: {raw_message[:100]}...")
            continue
            
        if speaker == 'dad':
            messages.append(content)
        else:
            if messages:  # If we have collected dad's messages
                training_pairs.append({
                    'text': f"### Human: How would my dad respond?\n### Assistant: {' '.join(messages)}"
                })
                messages = []
    
    # Don't forget the last group of messages
    if messages:
        training_pairs.append({
            'text': f"### Human: How would my dad respond?\n### Assistant: {' '.join(messages)}"
        })
    
    return training_pairs

def main():
    # Load and process conversations
    df = load_conversations('data/dad_conversations.csv')
    training_pairs = create_training_pairs(df)
    
    # Print some statistics
    print(f"Processed {len(df)} messages")
    print(f"Created {len(training_pairs)} training pairs")
    
    if len(training_pairs) == 0:
        print("Warning: No training pairs were created!")
    else:
        # Print a sample training pair to verify format
        print("\nSample training pair:")
        print(training_pairs[0]['text'][:200] + "...")
    
    # Save processed data
    with open('data/training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 