#Student Name: Abhishek Maheshkumar Parikh
#Student ID: A00075992

# Grading System


from datetime import datetime
from tabulate import tabulate

def calculate_overall_score(score):   # Function to calculate overall score
    score = int(c1 * 0.10 + c2 * 0.20 + c3 * 0.30 + fe * 0.40)   # Calculate overall score based on weights
    
    return score

def determine_category(score):                        # Determine category based on overall score
    # Adjust score ranges
    score_adjustments = [(100, 100), (82, 92), (72, 78), (62, 68), # List of score ranges
                      (52, 58),(42, 48),(32, 38),(5, 25),(0, 0)]

    categories = [(100, 100, "- Aurum Standard"), (82, 92, "- Upper First"), (72, 78, "- First"),  # list of categories
                  (62, 68, "- 2:1"), (52, 58, "- 2:2"), (42, 48, "- Third"), (32, 38, "- Condonable Fail"),
                  (5, 25, "- Fail"), (0, 0, "- Defecit Opus")]
    for  low, high in score_adjustments:
         if low <= score <= high:
            score = high
            break
   

    for low, high, category in categories:
        if low <= score <= high:
            return score, category

def main():
    print("Welcome to the student Grading System First, let's set up the module configuration.")

    def new_module():
        Do_you_want_to_configure_new_module = input("configure new module:(yes/No): ").lower()

        if Do_you_want_to_configure_new_module == "yes":
           module_name = input("Enter module name: ")
           components = int(input("How many assessment components does this (module_name) have?: "))
           components_list = []
           total_weight = 0
           for _ in range(components):
                component_name = input("Enter component name: ")
                weight = int(input("Enter component weight(%): "))
                total_weight += weight
                components_list.append((component_name, weight))
           if total_weight != 100:
            print("The total weight of all components must equal 100%. Please reconfigure the module.")
            return None
           else:
            return {"module_name": module_name, "components": components_list}
        else:
            return {"module_name": "Default Module", "components": []}
    
    module_config = new_module()
    print(module_config)  # Accessing module_config to avoid the undefined error

    students = []
    while len(students) < 3:
        Student_id = input("Enter student ID (2 digit) or 'end' to finish: ").strip()
        if Student_id.lower() == "end":
            break
        elif not Student_id.isdigit() or len(Student_id) != 2:
            print("The input you entered was invalid")
            continue

        name = input("enter student name: ")

        try:
            DOB = input("Enter date of birth(yyyy-mm-dd): ")  
            DOB = datetime.strptime(DOB, "%Y-%m-%d").date()
        except ValueError:
            print("The input you entered was invalid")
            continue
        age = datetime.now().year - DOB.year - ((datetime.now().month, datetime.now().day) < (DOB.month, DOB.day))
        

            
        try:
            c1 = int(input("Enter coursework score1 marks: "))
            c2 = int(input("Enter coursework score2 marks: "))
            c3 = int(input("Enter coursework score3 marks: "))
            fe = int(input("Enter final exam marks: "))
        except ValueError:
            print("The input you entered was invalid")
            continue

        raw_score = calculate_overall_score(c1, c2, c3, fe)
        overallscore, category = determine_category(raw_score)

        students.append({"ID": Student_id, "Name": name, "DOB": DOB, "Age": age, "Raw Score": raw_score, "Overall Score": overallscore, "Category": category})
        students.sort(key=lambda x: x[0])
        print(tabulate(students, headers="keys", tablefmt="grid"))

def advanced(filename, weight):
    students = []

    with open(filename, 'r') as file:
        for line in file:
            id, name, dob = line.strip().split(',')
            dob = datetime.strptime(dob, "%Y-%m-%d").date()
            age = datetime.now().year - dob.year - ((datetime.now().month, datetime.now().day) < (dob.month, dob.day))
            raw_score = calculate_overall_score(c1, c2, c3, fe)
            score, category = determine_category(raw_score)
            students.append({"ID": id, "Name": name, "DOB": dob, "Age": age, "raw": raw_score,"roundedscore": overallscore, "Category": category})

    students.sort(key=lambda x: x["0"])

    header = ["ID", "Name", "DOB", "Age", "Raw Score", "Overall Score", "Category"]
    print(tabulate(students, headers=header, tablefmt="grid"))

    with open('output.txt', 'w') as outfile:
        outfile.write(tabulate(students, headers=header, tablefmt="grid"))  
    print("Results have been written to output.txt")

if __name__ == "__main__":
    main()



   
    


        
    
    
