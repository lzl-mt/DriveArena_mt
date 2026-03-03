import time
import requests
import json


def send_request(server, uttid, parse_thing):
    url = server
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "max_tokens": 256,
        "temperature": 0.0,
        "prompt": parse_thing
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Response for {uttid}:")
    print(response.text)


def main(input_file, server):
    send_request(server, "test", input_file)
    # with open(input_file, 'r') as file:
    #     for line in file:
    #         parts = line.strip().split(' ', 1)
    #         if len(parts) != 2:
    #             print(f"Skipping invalid line: {line}")
    #             continue
    #         uttid, parse_thing = parts
    #         time.sleep(1)
    #         print("")
    #         print(uttid)
    #         print(parse_thing)
    #         send_request(server, uttid, parse_thing)


if __name__ == "__main__":
    input_file = "12345"
    server = "http://172.31.208.10:10019/generate"
    main(input_file, server)
