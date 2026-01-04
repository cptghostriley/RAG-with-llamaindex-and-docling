while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = _query_engine.query(query)
    
    # Stream the response token by token
    print("Response: ", end="", flush=True)
    for token in response.response_gen:
        print(token, end="", flush=True)
    print() 