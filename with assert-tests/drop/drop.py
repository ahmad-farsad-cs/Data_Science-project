def analyze_log(filename):
    hosts_requests = {}

    with open(filename, 'r') as file:
        for line in file:
            hostname = line.split()[0]

            if hostname in hosts_requests:
                hosts_requests[hostname] += 1
            else:
                hosts_requests[hostname] = 1

        output_filename = "records_" + filename

        with open(output_filename, 'w') as output_file:
            for host, count in hosts_requests.items():
                output_file.write(f"{host} {count}\n")


filename = input("Enter the filename: ")

analyze_log(filename)
