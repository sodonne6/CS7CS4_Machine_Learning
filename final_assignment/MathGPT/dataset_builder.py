## this file is for generating the maths dataset
import random
import os
import csv
import sympy as sp


#need some flags for dataset generation - what operations to include
ADD_SUB = True
MUL_DIV = True
INTEGRALS = False
DERIVATIVES = False

OP="add_sub_div_mul" #name of operation for output files
    
SEED = 1337
ADD_SUB_MIN, ADD_SUB_MAX = 0, 99
MUL_DIV_MIN, MUL_DIV_MAX = -45, 45  # much easier space
TEST_SET = 0.2 #percentage of data to reserve for testing

OUT_DIR = "add_sub_div_mul/add_sub_div_mul_dataset" #directory to save dataset to

def generate_add_sub(min_val, max_val,test_set):
   train_lines = []
   test_rows = []
   
   for i in range(min_val,max_val+1):
       for j in range (min_val, max_val+1):
           
           #integer answer of add 
           y_add = i + j
           add_eq = f"{i}+{j}={y_add}\n"
           prompt_add = f"{i}+{j}="
           answer_add = str(y_add)
           
           if random.random() < test_set:
               test_rows.append((prompt_add, answer_add,"+"))
           else:
               train_lines.append(add_eq)
        
           #subtraction - allow negative answers
           
           y_sub = i-j
           sub_eq = f"{i}-{j}={y_sub}\n"
           prompt_sub = f"{i}-{j}="
           answer_sub = str(y_sub)
           if random.random() < test_set:
                test_rows.append((prompt_sub, answer_sub,"-"))
           else:
                train_lines.append(sub_eq)
                    
   return train_lines, test_rows

def generate_mul_div(min_val, max_val,test_set):
    train_lines = []
    test_rows = []
    
    for i in range(min_val,max_val+1):
        for j in range (min_val, max_val+1):
            #multiplication
            y_mul = i * j
            mul_eq = f"{i}*{j}={y_mul}\n"
            prompt_mul = f"{i}*{j}="
            answer_mul = str(y_mul)
            if random.random() < test_set:
                test_rows.append((prompt_mul, answer_mul,"*"))
            else:
                train_lines.append(mul_eq)
                
            #division - only integer results, avoid div by 0
            if j != 0:          #to change back to no decimals put this back in and i % j == 0
                #y_div only to 2 decimal places
                y_div = round(i / j, 2)
                div_eq = f"{i}/{j}={y_div}\n"
                prompt_div = f"{i}/{j}="
                answer_div = str(y_div)
                if random.random() < test_set:
                    test_rows.append((prompt_div, answer_div,"/"))
                else:
                    train_lines.append(div_eq)
    
    #to be implemtted
    return train_lines, test_rows

#helping function to get correct answer for integrals and derivatives

        
   


def generate_integrals(low_bound=-2, high_bound=5,test_set=0.2,max_degree=3):
    """closed integrals only to keep answers as integers
    In the form INT[1,3] x^2 dx = 26"""
    
    train_lines = []
    test_rows = []
    
    for i in range(low_bound, high_bound+1):
        for j in range(low_bound, high_bound+1):
            if i == j:
                continue
            n = random.randint(1,max_degree)
            
            #integral of x^n from i to j
            integral_numerator = (j**(n+1) - i**(n+1))
            
            if integral_numerator % (n+1) != 0:
                continue #skip non integer results
            
            integral_value = integral_numerator // (n+1)
            integral_eq = f"INT[{i},{j}] x^{n} dx = {integral_value}\n"
            prompt_integral = f"INT[{i},{j}] x^{n} dx ="
            answer_integral = str(integral_value)
            if random.random() < test_set:
                test_rows.append((prompt_integral, answer_integral,"INT"))
            else:
                train_lines.append(integral_eq)
                
            
            
    
    #to be implemtted
    return train_lines, test_rows


def generate_derivatives(min_val, max_val,test_set): 
    """same format as integrals - keep it simple and closed form
    DERIV[x^2,3] = 6"""   
    
    #to be implemtted
    return                

                
        

def build_dataset(add_sub_min, add_sub_max, mul_div_min, mul_div_max, test_set):
    #combine all selectect ops to create one dataset
    
    all_train_lines = []
    all_test_rows = []
    
    if ADD_SUB:
        train_lines, test_rows = generate_add_sub(ADD_SUB_MIN, ADD_SUB_MAX,test_set)
        all_train_lines.extend(train_lines)
        all_test_rows.extend(test_rows)
    if MUL_DIV:
        train_lines, test_rows = generate_mul_div(MUL_DIV_MIN, MUL_DIV_MAX,test_set)
        all_train_lines.extend(train_lines)
        all_test_rows.extend(test_rows)
    if INTEGRALS:
        train_lines, test_rows = generate_integrals(
            low_bound=-3,
            high_bound=10,
            test_set=test_set,
            max_degree=4
        )
        all_train_lines.extend(train_lines)
        all_test_rows.extend(test_rows)
        
    if DERIVATIVES:
        train_lines, test_rows = generate_derivatives(add_sub_min, add_sub_max,test_set)
        all_train_lines.extend(train_lines)
        all_test_rows.extend(test_rows)
        
    #shuffle training so its mixed up
    random.shuffle(all_train_lines)
    
    return all_train_lines, all_test_rows

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_lines, test_rows = build_dataset(ADD_SUB_MIN, ADD_SUB_MAX, MUL_DIV_MIN, MUL_DIV_MAX, TEST_SET)
    
        
    train_path = os.path.join(OUT_DIR,f"maths_{OP}_train.txt")
    test_path = os.path.join(OUT_DIR,f"maths_{OP}_test.tsv")
    
    #write training file
    with open(train_path,"w",encoding="utf-8") as f:
        f.writelines(train_lines)
        
    #write tsv prompt/answer/operation test file
    with open(test_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["prompt","answer","operation"])
        for prompt, answer, operation in test_rows:
            writer.writerow([prompt, answer, operation])
        
    print(f"Wrote {len(train_lines)} training equations to {train_path}")
    print(f"Wrote {len(test_rows)}  test equations to {test_path}")
    #print(f"Operands range: [{MIN_VAL}, {MAX_VAL}], test fraction: {TEST_SET}")
    print(f"Ops enabled: {OP}")   
    
    
if __name__ == "__main__":
    main()
                 