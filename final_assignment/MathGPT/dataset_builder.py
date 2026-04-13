## this file is for generating the maths dataset
import random
import os
import csv


#need some flags for dataset generation - what operations to include
ADD_SUB = True
MUL_DIV = True
INTEGRALS = False
DERIVATIVES = False
TWOADD_MUL = True
THREEADD = True

OP="add_sub_div_mul_3terms" #name of operation for output files
    
SEED = 1337
ADD_SUB_MIN, ADD_SUB_MAX = -99, 99
MUL_DIV_MIN, MUL_DIV_MAX = -30, 30  # much easier space
TEST_SET = 0.2 #percentage of data to reserve for testing

OUT_DIR = "all_operations_balanced/dataset/all_ops_3terms" #directory to save dataset to

def generate_add_sub(min_val, max_val):
    examples = []
    for i in range(min_val, max_val + 1):
        for j in range(min_val, max_val + 1):
            #add
            y_add = i + j
            examples.append((f"({i}) + ({j}) =", str(y_add), "+"))

            #sub
            y_sub = i - j
            examples.append((f"({i}) - ({j}) =", str(y_sub), "-"))
    return examples




def generate_mul_div(mul_min=-99, mul_max=99, quot_min=-120, quot_max=120, div_min=1, div_max=20):
    examples = []

    #multiplication
    for i in range(mul_min, mul_max + 1):
        for j in range(mul_min, mul_max + 1):
            y_mul = i * j
            if keep_answer(y_mul):
                examples.append((f"({i}) * ({j}) =", str(y_mul), "*"))

     #division (exact integer quotients) - change to all answers later on maybe
    for q in range(quot_min, quot_max + 1):
        for d in range(div_min, div_max + 1):
            dividend = q * d
            divisor = d
            if keep_answer(q) and keep_answer(dividend):
                examples.append((f"({dividend}) / ({divisor}) =", str(q), "/"))

    return examples

#helping function to get correct answer for integrals and derivatives

def generate_two_addsub_one_mul(n_examples_per_pattern=10000):
    """Randomly sample expressions with two add/sub and one mul/div."""
    examples = []
    while len(examples) < 2 * n_examples_per_pattern:
        i = random.randint(ADD_SUB_MIN, ADD_SUB_MAX)
        j = random.randint(ADD_SUB_MIN, ADD_SUB_MAX)
        k = random.choice([x for x in range(-15,16) if x !=0])  #avoid zero multiplication

        y_add = (i + j) * k
        if keep_answer(y_add):
            examples.append((f"(({i}) + ({j})) * ({k}) =", str(y_add), "+*"))


        y_sub = (i - j) * k
        if keep_answer(y_sub):
            examples.append((f"(({i}) - ({j})) * ({k}) =", str(y_sub), "-*"))
    return examples

def generate_three_adds(n_examples=10000):
    """Randomly sample expressions with three additions."""
    examples = []
    for _ in range(n_examples):
        i = random.randint(ADD_SUB_MIN, ADD_SUB_MAX)
        j = random.randint(ADD_SUB_MIN, ADD_SUB_MAX)
        k = random.randint(ADD_SUB_MIN, ADD_SUB_MAX)

        y_add = i + j + k
        if keep_answer(y_add):
            examples.append((f"({i}) + ({j}) + ({k}) =", str(y_add), "++"))
        y_sub = i - j - k
        if keep_answer(y_sub):
            examples.append((f"({i}) - ({j}) - ({k}) =", str(y_sub), "--"))

    return examples
  

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


def no_duplicates(examples):
    seen = set()
    out = []
    for prompt, answer, op in examples:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, answer, op))
    return out
            
def train_test_split(examples, test_frac):
    random.shuffle(examples)
    split_index = int(len(examples) * (1 - test_frac))
    train_rows = examples[:split_index]
    test_rows = examples[split_index:]
    return train_rows, test_rows    

def balance_operations(examples, n):
    if n <= len(examples):
        return random.sample(examples, n)
    return [random.choice(examples) for _ in range(n)]  #oversample

MAX_ANSWER_LEN = 5

def keep_answer(y, max_len=MAX_ANSWER_LEN):
    return len(str(y)) <= max_len



def build_dataset(test_frac):
    pools = {}

    if ADD_SUB:
        addsub = generate_add_sub(ADD_SUB_MIN, ADD_SUB_MAX)
        pools["+"] = no_duplicates([ex for ex in addsub if ex[2] == "+"])
        pools["-"] = no_duplicates([ex for ex in addsub if ex[2] == "-"])

    if MUL_DIV:
        muldiv = generate_mul_div(
            mul_min=MUL_DIV_MIN, mul_max=MUL_DIV_MAX,
            quot_min=-90, quot_max=90,
            div_min=1, div_max=12
        )
        pools["*"] = no_duplicates([ex for ex in muldiv if ex[2] == "*"])
        pools["/"] = no_duplicates([ex for ex in muldiv if ex[2] == "/"])
    if TWOADD_MUL:
        twoadd_muldiv = generate_two_addsub_one_mul()
        pools["+*"] = no_duplicates([ex for ex in twoadd_muldiv if ex[2] == "+*"])
        pools["-*"] = no_duplicates([ex for ex in twoadd_muldiv if ex[2] == "-*"])
    if THREEADD:
        three_adds = generate_three_adds()
        pools["++"] = no_duplicates([ex for ex in three_adds if ex[2] == "++"])
        pools["--"] = no_duplicates([ex for ex in three_adds if ex[2] == "--"])


    #remove empty pools
    pools = {op: pool for op, pool in pools.items() if len(pool) > 0}
    
    train_rows = []
    test_rows  = []

    for op, pool in pools.items():
        random.shuffle(pool)

        n_test = max(1, int(len(pool) * test_frac))
        op_test = pool[:n_test]
        op_train = pool[n_test:]  


        if len(op_train) == 0:
            # steal 1 back from test
            op_train = op_test[:1]
            op_test  = op_test[1:]

        train_rows.extend(op_train)
        test_rows.extend(op_test)

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    train_lines = [f"{p}{a}\n" for (p, a, op) in train_rows]
    
    
    return train_lines, test_rows

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_lines, test_rows = build_dataset(TEST_SET)
    
        
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
        
    print(f"Wrote {len(train_lines)}")
    print(f"Wrote {len(test_rows)}")
     
    
    
if __name__ == "__main__":
    main()
                 