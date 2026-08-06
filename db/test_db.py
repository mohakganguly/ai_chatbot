from repository import (
    create_conversation,
    save_message,
    get_messages
)

thread_id = "test-thread"

create_conversation(
    thread_id,
    "Testing DB"
)

save_message(
    thread_id,
    "user",
    "Hello"
)

messages = get_messages(
    thread_id
)

for message in messages:

    print(
        message.role,
        message.content
    )