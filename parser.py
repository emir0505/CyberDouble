import json
import re

def anonymize_text(text):
    if isinstance(text, str):
        text = re.sub(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', '[CRYPTO_ADDRESS]', text)  # BTC
        text = re.sub(r'\b0x[a-fA-F0-9]{40}\b', '[CRYPTO_ADDRESS]', text)  # ETH/ERC20
        # API-токены (например, JWT или случайные хеши)
        text = re.sub(r'\b[a-fA-F0-9]{32}\b', '[API_TOKEN]', text)  # MD5 хеш
        text = re.sub(r'\b[a-fA-F0-9]{40}\b', '[API_TOKEN]', text)  # SHA-1
        text = re.sub(r'\b[a-fA-F0-9]{64}\b', '[API_TOKEN]', text)  # SHA-256
        # Стандартные токены доступа (Bearer, OAuth и т.д.)
        text = re.sub(r'\b(ey[a-zA-Z0-9_-]{20,}\.ey[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b', '[JWT_TOKEN]', text)  # JWT
        text = re.sub(r'\bBearer\s+[a-zA-Z0-9._-]+\b', '[AUTH_TOKEN]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'(?<!\w)@\w+', '[USERNAME]', text)
        # text = re.sub(r'https?:\/\/(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?', '[LINK]', text)
        text = re.sub(r'https?:\/\/[^\s]+', '[LINK]', text)
        text = re.sub(r'@\w+', '[USERNAME]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b(?:\+7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b', '[PHONE]', text)
        text = re.sub(r'\b\d{4}\s?\d{6}\b', '[PASSPORT]', text)
        text = re.sub(r'\b\d{12}\b', '[INN]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{3} \d{2}\b', '[SNILS]', text)
        text = re.sub(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', '[CARD]', text)
        text = re.sub(r'\b[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\s[А-ЯЁ][а-яё]+\b', '[FIO]', text)
        text = re.sub(r'\b(ул\.|улица|пр\.|проспект|д\.|дом|кв\.|квартира)\s[А-Яа-яё0-9\s\-]+\b', '[ADDRESS]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{2}\.\d{2}\.\d{4}\b', '[DATE]', text)
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
        return text
    elif isinstance(text, dict) and 'text' in text:
        text['text'] = anonymize_text(text['text'])
        return text['text']
    return text

def convert_to_jsonl(input_file, output_file, user_id, file_mode, min_messages=30):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, file_mode, encoding="utf-8") as outfile:
        for chat in data['chats']['list']:
            msg_type = chat.get('type', "")
            chat_name = chat.get('name')

            if msg_type != "personal_chat" or chat_name is None:
                continue

            messages = chat.get('messages', [])
            formatted_messages = []

            for message in messages:
                if 'from_id' not in message or 'text' not in message:
                    continue

                text = message.get('text', '')
                if not isinstance(text, str) or text == '':
                    continue

                text = anonymize_text(text)
                if not text:
                    continue

                formatted_messages.append({
                    "date": message.get('date', 'unknown'),
                    "from": message.get('from', 'unknown'),
                    "text": text
                })

            if len(formatted_messages) >= min_messages:
                # Формируем JSON-объект с отступами (не экранируем!)
                json_object = {
                    "name": chat_name,
                    "messages": formatted_messages
                }
                json_string = json.dumps(json_object, ensure_ascii=False, indent=1)  

                # Экранируем json_string, чтобы он корректно записался в "content"
                wrapped_json = json.dumps(
                    {
                        "sample": 
                        # [
                            {"role": "user", "content": json_string}
                        # ]
                    },
                    ensure_ascii=False
                )

                outfile.write(wrapped_json + '\n')

if __name__ == '__main__':
    for user in USERS:
        input_file = user["metadata"]["input_file"]
        output_file = user["metadata"]["output_file"]
        user_id = user["metadata"]["user_id"]
        file_mode = user["metadata"]["file_mode"]
        convert_to_jsonl(input_file, output_file, user_id, file_mode)