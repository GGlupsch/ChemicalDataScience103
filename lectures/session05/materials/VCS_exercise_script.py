def greetings(name: str) -> str:
    """Greetings by python"""
    return f"Hello {name}!"

if __name__ == "__main__":
    name = input("What is your name? ")
    print(greetings(name))
