def build_prompt(user_query: str, documents: list[str]) -> list[str]:
    content = "\n\n".join(documents)
    return [
        {
            'role': 'user',
            'content': ('Ты персональный ассистент для подготовки к экзаменам,' 
                        'твоя задача - привести ученика к правильному решению через наводящие вопросы' 
                        f'и ни в коем случае не писать решение сразу!\nВот похожие задачи с решением: {content}.\n'
                        f'Вот сам вопрос ученика: {user_query}.')
        }
    ]
