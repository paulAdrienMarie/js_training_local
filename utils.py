import json

def filter():
    # Define the file path
    cache_file_path = "./cache.json"

    # Open and load the cache JSON file
    with open(cache_file_path) as f:
        cache = json.load(f)

    # Extract values from the cache and filter them
    # Convert cache values to a list to use indexing
    values_list = list(cache.values())
    filtered_values = [value for value in values_list if value == "I don't know"]

    filtered_dict = {}

    for item in cache.items():
        if item[1] == "I don't know":
            pass

        filtered_dict[item[0]] = item[1]

    with open("filtered_dataset.json","w") as f:
        json.dump(filtered_dict, f)

    print("LENGTH OF CACHE = {}".format(len(cache)))
    print("NB OCCURENCE OF I DON'T KNOW = {}".format(len(filtered_values)))
    print("NB WELL PREDICTED = {}".format(len(cache)- len(filtered_values)))
    
if __name__ == "__main__":
    filter()
