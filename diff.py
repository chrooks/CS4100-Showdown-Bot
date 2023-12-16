def parse_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = {}
    current_label = None

    for line in lines:
        if 'results:' in line.lower():
            current_label = line.split(':')[0].strip()
            results[current_label] = {}
        elif 'Average of Average Performance:' in line or 'Average of Average Performances:' in line:
            results[current_label]['Average Performance'] = float(
                line.split(':')[-1].strip())
        elif 'Average of Standard Deviation:' in line:
            results[current_label]['Standard Deviation'] = float(
                line.split(':')[-1].strip())
        elif 'Average of Weighted Average Performance:' in line or 'Average of Weighted Average Performances:' in line:
            results[current_label]['Weighted Average Performance'] = float(
                line.split(':')[-1].strip())

    return results


def calculate_differences(results):
    baseline = results['Baseline results']
    for label, values in results.items():
        if label == 'Baseline results':
            continue
        print(f'\n{label} differences from Baseline:')
        for key, value in values.items():
            difference = value - baseline[key]
            print(f'{key}: {difference:+.4f}')


def main():
    file_path = 'res.txt'  # Replace with the path to your file
    results = parse_results(file_path)
    calculate_differences(results)


if __name__ == '__main__':
    main()