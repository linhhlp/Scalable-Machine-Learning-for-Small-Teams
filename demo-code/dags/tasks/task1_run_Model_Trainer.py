### FIRST TASK ############

def main():
    import requests
    result = requests.get("http://34.69.150.86/run")
    print(result.json())