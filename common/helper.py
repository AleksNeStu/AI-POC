def read_file(file_path):
    """
    Reads a file and returns its contents as a string.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Contents of the file as a string.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except Exception as e:
        return f"Error: {e}"