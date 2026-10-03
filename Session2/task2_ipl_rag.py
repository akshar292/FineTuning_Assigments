# Task 2: Simple RAG Workflow using IPL Stats

# Read IPL statistics from text file
with open("ipl_stats.txt", "r", encoding="utf-8") as file:
    stats = file.readlines()


# Take question from user
question = input("Ask a question about IPL: ")

# Convert question to lowercase
question_lower = question.lower()


# Retrieve relevant lines
retrieved_data = []

for line in stats:

    line_lower = line.lower()

    # Check if important words from question
    # are present in the line
    words = question_lower.split()

    for word in words:

        if len(word) > 3 and word in line_lower:
            if line not in retrieved_data:
                retrieved_data.append(line)
            break


# Display retrieved information
print("\nRetrieved Information:")

if retrieved_data:
    for data in retrieved_data:
        print(data.strip())
else:
    print("No relevant information found.")


# Generate a simple combined answer
print("\nFinal Answer:")

if retrieved_data:
    print("Based on the IPL statistics:")
    
    for data in retrieved_data:
        print("-", data.strip())

else:
    print("Sorry, I could not find relevant IPL statistics.")