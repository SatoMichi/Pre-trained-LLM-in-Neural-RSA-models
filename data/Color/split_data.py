import csv

def split_csv(input_file, output_prefix, chunk_size):
    with open(input_file, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        chunk_count = 1
        row_count = 0
        for row in reader:
            if row_count % chunk_size == 0:
                output_file = f"{output_prefix}_{chunk_count}.csv"
                with open(output_file, 'w', newline='') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(header)

            with open(output_file, 'a', newline='') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(row)
            row_count += 1
            if row_count % chunk_size == 0:
                chunk_count += 1

    print(f"Split complete. Total rows: {row_count}, Total chunks: {chunk_count - 1}")


# Usage example
input_file = "colors.csv"       # Path to your input CSV file
output_prefix = "new_colors"    # Prefix for the output file names
chunk_size = 10000              # Number of rows per output file

split_csv(input_file, output_prefix, chunk_size)
