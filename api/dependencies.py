from cryptography.fernet import Fernet

# Generate a key for encryption/decryption
key = Fernet.generate_key()
fernet = Fernet(key)

def encrypt_file(contents):
    """
    Encrypts file contents using the Fernet symmetric encryption algorithm.
    
    Parameters:
    - contents (bytes): The contents of the file to encrypt.
    
    Returns:
    - bytes: The encrypted contents.
    """
    return fernet.encrypt(contents)

def decrypt_file(encrypted_contents):
    """
    Decrypts encrypted file contents using the Fernet symmetric decryption algorithm.
    
    Parameters:
    - encrypted_contents (bytes): The encrypted contents of the file.
    
    Returns:
    - bytes: The decrypted contents.
    """
    return fernet.decrypt(encrypted_contents)
