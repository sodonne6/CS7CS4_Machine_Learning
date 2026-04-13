## this file is for generating the bool dataset

#need to modify to generate bool dataset
#AND, OR, NOT, XOR, NAND, NOR, XNOR, etc.
import random
import os
import csv


#need some flags for dataset generation - what operations to include
AND_OR = True
NAND_NOR = True
XOR_XNOR = True
MULTI_PROMPTS = True
INVERSE = True #include NOT versions of basic ops
SIX_INPUT = False #include some 6 input multi ops
SIX_TO_TEST = False #put 6 input multi ops in test set - if true send to stress UNSEEN_6 to test - if false send to stress_test instead - MULTI_6_INPUT to train always if SIX_INPUT enabled
RIGHT_UNBALANCE_TEST = True #include right unbalanced trees in test set
RIGHT_THEN_LEFT_TEST = True #include in test if true
UNSEEN_4_IN_TEST = True #put some unseen 4 input structures in test set

OP="4input" #name of operation for output files
    
SEED = 1337
#using different split logic now
#TEST_SET = 0.05 #percentage of data to reserve for testing

OUT_DIR = "basic_operations/dataset/4_ops_all_struct_types_unseen_in_test" #directory to save dataset to

def generate_and_or():
    """basic operations for AND and OR - all these should be in training set"""
    examples = []
    
    #need to create all combinations, basically a truth table
    
    for i in range(2):
        for j in range(2):
            # and
            y_and = int(i and j)
            examples.append((f"{i} AND {j} =", str(y_and), "AND"))

            # or
            y_or = int(i or j)
            examples.append((f"{i} OR {j} =", str(y_or), "OR"))
    return examples




def generate_nand_nor():
    """basic operations for NAND and NOR - all these should be in training set"""
    examples = []

    # nand
    for i in range(2):
        for j in range(2):
            y_nand = int(not (i and j))
            examples.append((f"{i} NAND {j} =", str(y_nand), "NAND"))

    # nor
    for i in range(2):
        for j in range(2):
            y_nor = int(not (i or j))
            examples.append((f"{i} NOR {j} =", str(y_nor), "NOR"))
    return examples

def generate_xor_xnor():
    """basic operations for XOR and XNOR - all be in training set"""
    examples = []
    # xor
    for i in range(2):
        for j in range(2):
            y_xor = int(i ^ j)
            examples.append((f"{i} XOR {j} =", str(y_xor), "XOR"))
    # xnor
    for i in range(2):
        for j in range(2):
            y_xnor = int(not (i ^ j))
            examples.append((f"{i} XNOR {j} =", str(y_xnor), "XNOR"))
            
    return examples

#helping function to get correct answer for integrals and derivatives

def generate_inverse_operations():
    """this will have basic gates but with the ~ symbol to indicate NOT operation
    for example NOT(A AND B) - NOT(1 AND 0) = 1 - these can be a little tricky as they have two steps involved
    split between train and test as we already have nand and nor in the basic set"""
    examples = []
    # NOT AND (NAND)
    for i in range(2):
        for j in range(2):
            y_nand = int(not (i and j))
            examples.append((f"NOT({i} AND {j}) =", str(y_nand), "NOT"))
    # NOT OR (NOR)
    for i in range(2):
        for j in range(2):
            y_nor = int(not (i or j))
            examples.append((f"NOT({i} OR {j}) =", str(y_nor), "NOT"))
    for i in range(2):
        y_not = int(not i)
        examples.append((f"NOT({i}) =", str(y_not), "NOT"))
    
    return examples

OPERATORS = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]

#helper function to compute operations instead of writing each time
def op_eval(op, x, y):
    """cycle through each possible op and compute"""
    if op == "AND":  
        return x & y
    if op == "OR":   
        return x | y
    if op == "XOR":  
        return x ^ y
    if op == "XNOR": 
        return 1 - (x ^ y)
    if op == "NAND": 
        return 1 - (x & y)
    if op == "NOR":  
        return 1 - (x | y)

def multiple_operations_4_input():
    """Generate multi-step boolean expressions with many prompt templates - have a variety of different formations with the rule that 
    there can only be 4 inputs"""
    examples = []

    # operator surface forms (same semantics, different text)
    AND_FORMS  = ["AND"]
    OR_FORMS   = ["OR"]
    XOR_FORMS  = ["XOR"]
    NOT_FORMS  = ["NOT"]   # unary
    NAND_FORMS = ["NAND"]       # keep word form (could add "!AND" etc later)
    NOR_FORMS  = ["NOR"]        # keep word form

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):

                    # precompute base ops (ints 0/1)
                    ab_and  = (a and b)
                    ab_or   = (a or b)
                    ab_xor  = (a ^ b)
                    cd_and  = (c and d)
                    cd_or   = (c or d)
                    cd_xor  = (c ^ d)

                    ab_nand = (not (a and b))
                    ab_nor  = (not (a or b))
                    cd_nand = (not (c and d))
                    cd_nor  = (not (c or d))
                    ab_xnor = not (a ^ b)
                    cd_xnor = not (c ^ d)
                    

                    #(A AND B) OR (C NAND D)
                    for AND in AND_FORMS:
                        for OR in OR_FORMS:
                            for NAND in NAND_FORMS:
                                val = (ab_and) or (cd_nand)
                                examples.append((f"({a} {AND} {b}) {OR} ({c} {NAND} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #(A OR B) AND (C NOR D)
                    for OR in OR_FORMS:
                        for AND in AND_FORMS:
                            for NOR in NOR_FORMS:
                                val = (ab_or) and (cd_nor)
                                examples.append((f"({a} {OR} {b}) {AND} ({c} {NOR} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #(A NOR B) XOR (C AND D)
                    for NOR in NOR_FORMS:
                        for XOR in XOR_FORMS:
                            for AND in AND_FORMS:
                                val = (ab_nor) ^ (cd_and)
                                examples.append((f"({a} {NOR} {b}) {XOR} ({c} {AND} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #~NOT(A AND B) OR (C XOR D)
                    for NOT in NOT_FORMS:
                        for AND in AND_FORMS:
                            for OR in OR_FORMS:
                                for XOR in XOR_FORMS:
                                    val = (not (ab_and)) or (cd_xor)
                                    # two equivalent surface forms for NOT:
                                    examples.append((f"{NOT}({a} {AND} {b}) {OR} ({c} {XOR} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #(A XOR B) AND ~(C OR D)
                    for XOR in XOR_FORMS:
                        for AND in AND_FORMS:
                            for NOT in NOT_FORMS:
                                for OR in OR_FORMS:
                                    val = (ab_xor) and (not (cd_or))
                                    examples.append((f"({a} {XOR} {b}) {AND} {NOT}({c} {OR} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #(A NAND B) XOR (C NOR D)
                    for NAND in NAND_FORMS:
                        for XOR in XOR_FORMS:
                            for NOR in NOR_FORMS:
                                val = (ab_nand) ^ (cd_nor)
                                examples.append((f"({a} {NAND} {b}) {XOR} ({c} {NOR} {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    #~(A XOR B) OR (C AND D)
                    for NOT in NOT_FORMS:
                        for XOR in XOR_FORMS:
                            for OR in OR_FORMS:
                                for AND in AND_FORMS:
                                    val = (not (ab_xor)) or (cd_and)
                                    examples.append((f"{NOT}({a} {XOR} {b}) {OR} ({c} {AND} {d}) =", str(int(val)), "MULTI_4_INPUT"))
                    # (A OR B) OR (C OR D)
                    for OR in OR_FORMS:
                        val = (ab_or) or (cd_or)
                        examples.append((f"({a} {OR} {b}) {OR} ({c} {OR} {d}) =", str(int(val)), "MULTI_4_INPUT"))
                        
                    # (A AND B) AND (C OR D)
                    val = (ab_and) and (cd_or)
                    examples.append((f"({a} AND {b}) AND ({c} OR {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A OR B) OR (C AND D)
                    val = (ab_or) or (cd_and)
                    examples.append((f"({a} OR {b}) OR ({c} AND {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A XOR B) OR (C NOR D)
                    val = (ab_xor) or (cd_nor)
                    examples.append((f"({a} XOR {b}) OR ({c} NOR {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A XNOR B) AND (C XOR D)
                    val = (ab_xnor) and (cd_xor)
                    examples.append((f"({a} XNOR {b}) AND ({c} XOR {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # NOT((A OR B) AND (C OR D))
                    val = not ((ab_or) and (cd_or))
                    examples.append((f"NOT(({a} OR {b}) AND ({c} OR {d})) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A AND NOT(B)) OR (C AND D)
                    val = (a and (not b)) or (cd_and)
                    examples.append((f"({a} AND NOT({b})) OR ({c} AND {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A OR NOT(B)) AND (C XOR D)
                    val = (a or (not b)) and (cd_xor)
                    examples.append((f"({a} OR NOT({b})) AND ({c} XOR {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    # (A NAND B) OR (C NAND D)
                    val = (ab_nand) or (cd_nand)
                    examples.append((f"({a} NAND {b}) OR ({c} NAND {d}) =", str(int(val)), "MULTI_4_INPUT"))

                    
                    

    return examples

def four_input_ops_random_balanced_tree():
    """generate random 4 input boolean expressions too boost dataset size - loop inside the function 
    to generate (1/0 OP 1/0) OP (1/0 OP 1/0) =	1/0"""
    examples = []
    pattern_count = 0
    OPERATORS = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]
    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
            #for each (a,b,c,d) there are 6^3 = 216 unique operator combos
                pattern_count +=1
                if pattern_count % 5 ==0:
                    tag = "HOLDOUT_4_INPUT"
                else:
                    tag = "MULTI_4_INPUT"
                    
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):

                    

                                #compute left side
                                if op1 == "AND":
                                    left = a & b
                                elif op1 == "OR":
                                    left = a | b
                                elif op1 == "NAND":
                                    left = 1 - (a & b)
                                elif op1 == "NOR":
                                    left = 1 - (a | b)
                                elif op1 == "XOR":
                                    left = a ^ b
                                elif op1 == "XNOR":
                                    left = 1 - (a ^ b)

                                #compute right side
                                if op2 == "AND":
                                    right = c & d
                                elif op2 == "OR":
                                    right = c | d
                                elif op2 == "NAND":
                                    right = 1 - (c & d)
                                elif op2 == "NOR":
                                    right = 1 - (c | d)
                                elif op2 == "XOR":
                                    right = c ^ d
                                elif op2 == "XNOR":
                                    right = 1 - (c ^ d)

                                #combine
                                if op3 == "AND":
                                    val = int(left & right)
                                elif op3 == "OR":
                                    val = int(left | right)
                                elif op3 == "XOR":
                                    val = int(left ^ right)
                                elif op3 == "NAND":
                                    val = int(1 - (left & right))
                                elif op3 == "NOR":
                                    val = int(1 - (left | right))
                                elif op3 == "XNOR":
                                    val = int(1 - (left ^ right))

                                examples.append((f"({a} {op1} {b}) {op3} ({c} {op2} {d}) =", str(int(val)), tag))

    return examples

def four_input_ops_random_right_unbalanced_tree_with_not():
    """generate 4 input boolean expressions with a right-branching tree
    to generate a OP (b OP (c OP d)) = 1/0, and also NOT(a OP (b OP (c OP d))) = 1/0
    a op1 (b op2 (c op3 d)
    maybe change so that if toggled then this goes to unseen bacuse i think i have exhausted all the unique structures for 4 input equations"""
    examples = []
    OPERATORS = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]
    pattern_count = 0
    
    if RIGHT_UNBALANCE_TEST:
        print("right unbalanced going to unseen")
    
    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
    #for each (a,b,c,d) there are 6^3 = 216 unique operator combos
                pattern_count +=1
                if pattern_count % 5 ==0:
                    #maybe swap so this is in unseen test so theres a pattern which isnt dominated by not statements
                    if RIGHT_UNBALANCE_TEST:
                        tag = "UNSEEN_4"
                    else:
                        tag = "HOLDOUT_4_INPUT"
                else:
                    if RIGHT_UNBALANCE_TEST:
                        tag = "UNSEEN_4"
                    else: 
                        tag = "MULTI_4_INPUT"
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):

                    

                                #compute inner (c op3 d)
                                if op3 == "AND":
                                    inner = c & d
                                elif op3 == "OR":
                                    inner = c | d
                                elif op3 == "NAND":
                                    
                                    inner = 1 - (c & d)
                                elif op3 == "NOR":
                                    inner = 1 - (c | d)
                                elif op3 == "XOR":
                                    inner = c ^ d
                                elif op3 == "XNOR":
                                    
                                    inner = 1 - (c ^ d)

                                #compute middle (b op2 inner)
                                if op2 == "AND":
                                    middle = b & inner
                                elif op2 == "OR":
                                    middle = b | inner
                                elif op2 == "NAND":
                                    middle = 1 - (b & inner)
                                elif op2 == "NOR":
                                    middle = 1 - (b | inner)
                                elif op2 == "XOR":
                                    middle = b ^ inner
                                    
                                elif op2 == "XNOR":
                                    middle = 1 - (b ^ inner)
                                    

                                #compute final (a op1 middle)
                                if op1 == "AND":
                                    val = int(a & middle)
                                    
                                elif op1 == "OR":
                                    val = int(a | middle)
                                elif op1 == "XOR":
                                    val = int(a ^ middle)
                                    
                                elif op1 == "NAND":
                                    val = int(1 - (a & middle))
                                    
                                elif op1 == "NOR":
                                    val = int(1 - (a | middle))
                                elif op1 == "XNOR":
                                    val = int(1 - (a ^ middle))

                                #plain version
                                examples.append((f"{a} {op1} ({b} {op2} ({c} {op3} {d})) =", str(int(val)), tag))

                                #NOT wrapped version
                                
                                val_not = int(1 - val)
                                examples.append((f"NOT({a} {op1} ({b} {op2} ({c} {op3} {d}))) =", str(int(val_not)), tag))

    return examples


def four_input_ops_right_then_left_tree():
    """a op1 ((b op2 c) op3 d) = 1/0"""
    examples = []
    pattern_count = 0
    
    if RIGHT_THEN_LEFT_TEST:
        print("right then left going to unseen")
    
    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
                pattern_count +=1
                if pattern_count % 5 ==0:
                    if RIGHT_THEN_LEFT_TEST:
                        tag = "UNSEEN_4"
                        #print("right then left going to unseen")
                    else:
                        tag = "HOLDOUT_4_INPUT"
                else:
                    if RIGHT_THEN_LEFT_TEST:
                        tag = "UNSEEN_4"
                        #print("right then left going to unseen")
                    else:
                        tag = "MULTI_4_INPUT"
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):
                                
                                #compute b c first
                                if op2 == "AND":
                                    val1 = b & c
                                elif op2 == "OR":
                                    val1 = b | c
                                elif op2 == "NAND":
                                    val1 = 1 - (b & c)
                                elif op2 == "NOR":
                                    val1 = 1 - (b | c)
                                elif op2 == "XOR":
                                    val1 = b ^ c
                                elif op2 == "XNOR":
                                    val1 = 1 - (b ^ c)

                                #compute first part with d
                                if op3 == "AND":
                                    val2 = val1 & d
                                elif op3 == "OR":
                                    val2 = val1 | d
                                elif op3 == "NAND":
                                    val2 = 1 - (val1 & d)
                                elif op3 == "NOR":
                                    val2 = 1 - (val1 | d)
                                elif op3 == "XOR":
                                    val2 = val1 ^ d
                                elif op3 == "XNOR":
                                    val2 = 1 - (val1 ^ d)
                                    
                                #take val 2 with a to finish
                                if op1 == "AND":
                                    val = int(a & val2)
                                elif op1 == "OR":
                                    val = int(a | val2)
                                elif op1 == "XOR":
                                    val = int(a ^ val2)
                                elif op1 == "NAND":
                                    val = int(1 - (a & val2))
                                elif op1 == "NOR":
                                    val = int(1 - (a | val2))
                                elif op1 == "XNOR":
                                    val = int(1 - (a ^ val2))
                                
                                examples.append((f"{a} {op1} (({b} {op2} {c}) {op3} {d}) =", str(int(val)), tag))
                                
    return examples
                                
                    


def four_input_ops_random_unbalanced_tree():
    """generate random 4 input boolean expressions too boost dataset size - loop inside the function 
    to generate ((1/0 OP 1/0) OP 1/0) OP 1/0) =	1/0
    ((a {op1} b) op2 c) op3 d ="""
    examples = []
    OPERATORS = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]
    pattern_count = 0
    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
    #for each (a,b,c,d) there are 6^3 = 216 unique operator combos
                pattern_count +=1
                if pattern_count % 5 ==0:
                    tag = "HOLDOUT_4_INPUT"
                else:
                    tag = "MULTI_4_INPUT"
                    
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):

                    

                                #compute left side
                                if op1 == "AND":
                                    left = a & b
                                elif op1 == "OR":
                                    left = a | b
                                elif op1 == "NAND":
                                    left = 1 - (a & b)
                                elif op1 == "NOR":
                                    left = 1 - (a | b)
                                elif op1 == "XOR":
                                    left = a ^ b
                                elif op1 == "XNOR":
                                    left = 1 - (a ^ b)

                                #compute middle
                                if op2 == "AND":
                                    middle = left & c
                                elif op2 == "OR":
                                    middle = left | c
                                elif op2 == "NAND":
                                    middle = 1 - (left & c)
                                elif op2 == "NOR":
                                    middle = 1 - (left | c)
                                elif op2 == "XOR":
                                    middle = left ^ c
                                elif op2 == "XNOR":
                                    middle = 1 - (left ^ c)

                                #compute right side   
                                if op3 == "AND":
                                    val = int(middle & d)
                                elif op3 == "OR":
                                    val = int(middle | d)
                                elif op3 == "XOR":
                                    val = int(middle ^ d)
                                elif op3 == "NAND":
                                    val = int(1 - (middle & d))
                                elif op3 == "NOR":
                                    val = int(1 - (middle | d))
                                elif op3 == "XNOR":
                                    val = int(1 - (middle ^ d))

                                examples.append((f"(({a} {op1} {b}) {op2} {c}) {op3} {d} =", str(int(val)), tag))

    return examples

def four_input_ops_mixed_bracketing():
    """"(a op1 (b op2 c)) op3 d ="""
    examples = []
    pattern_count = 0

    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
                pattern_count += 1
                
                if (pattern_count % 5 == 0):
                    tag = "HOLDOUT_4_INPUT" 
                else:
                    tag = "MULTI_4_INPUT"

                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):
                                inner = op_eval(op2, b, c)
                                left  = op_eval(op1, a, inner)
                                val   = op_eval(op3, left, d)

                                examples.append(
                                    (f"({a} {op1} ({b} {op2} {c})) {op3} {d} =", str(val), tag)
                                )
    return examples

def four_input_ops_mixed_with_not_leaves():
    """(NOT({a}) op1 (b op2 c)) op3 NOT(d)"""
    examples = []
    pattern_count = 0

    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
                pattern_count += 1
                if (pattern_count % 5 == 0):
                    tag = "HOLDOUT_4_INPUT"
                else:
                    tag = "MULTI_4_INPUT"

                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):
                                inner = op_eval(op2, b, c)
                                left  = op_eval(op1, 1 - a, inner)   # NOT(a)
                                val   = op_eval(op3, left, 1 - d)    # NOT(d)

                                examples.append(
                                    (f"(NOT({a}) {op1} ({b} {op2} {c})) {op3} NOT({d}) =", str(val), tag)
                                )
    return examples

#need to give more examples of NOT in 4 input equations when there are 4 inputs - cant be the same as the unique structures for test

def for_input_ops_nested_not():
    examples = []
    pattern_count = 0

    for op1 in OPERATORS:
        for op2 in OPERATORS:
            for op3 in OPERATORS:
                pattern_count += 1
                if (pattern_count % 5 == 0):
                    tag = "HOLDOUT_4_INPUT"
                else:
                    tag = "MULTI_4_INPUT"
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            for d in range(2):
                                not_a = 1-a
                                not_b = 1-b
                                not_c = 1-c
                                not_d = 1-d
                                
                                cd = op_eval(op3, c, d)
                                ab = op_eval(op1, a, b)

                                #((a op1 NOT(b)) op2 NOT(c)) op3 d
                                p1 = op_eval(op1, a, not_b)
                                p2 = op_eval(op2, p1, not_c)
                                val = op_eval(op3, p2, d)
                                examples.append((f"(({a} {op1} NOT({b})) {op2} NOT({c})) {op3} {d} =", str(val), tag))
                                
                                #(a op1 (NOT(b) op2 c)) op3 d
                                p1 = op_eval(op2, not_b, c)
                                p2 = op_eval(op1, a, p1)
                                val = op_eval(op3, p2, d)
                                examples.append((f"({a} {op1} (NOT({b}) {op2} {c})) {op3} {d} =", str(val), tag))
                                
                                #((a op1 NOT(b)) op2 (NOT(c op3 d))) =
                                ab_mixed = op_eval(op1, a, 1 - b)
                                right = 1 - cd
                                val = op_eval(op2, ab_mixed, right)
                                examples.append((f"(({a} {op1} NOT({b})) {op2} (NOT({c} {op3} {d}))) =",str(int(val)),tag))
                                
                                #((NOT(a op1 b)) op2 (c op3 d)) =
                                left = 1 - ab
                                val = op_eval(op2, left, cd)
                                examples.append((f"((NOT({a} {op1} {b})) {op2} ({c} {op3} {d})) =",str(int(val)),tag))
                                
                                #NOT(((a op1 b) op2 (c op3 d))) =
                                inner = op_eval(op2, ab, cd)
                                val = 1 - inner
                                examples.append((f"NOT((({a} {op1} {b}) {op2} ({c} {op3} {d}))) =",str(int(val)),tag))
                                
    return examples
                                
                                

                            

def three_input_ops_left_branch():
    examples = []
    pattern_count = 0

    for op1 in OPERATORS:
        for op2 in OPERATORS:
            pattern_count += 1
            if (pattern_count % 5 == 0):
                tag = "HOLDOUT_3_INPUT"
            else:
                tag = "MULTI_3_INPUT"

            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        left = op_eval(op1, a, b)
                        val  = op_eval(op2, left, c)
                        examples.append((f"({a} {op1} {b}) {op2} {c} =", str(val), tag))
    return examples


def three_input_ops_right_branch_with_not():
    examples = []
    pattern_count = 0

    for op1 in OPERATORS:
        for op2 in OPERATORS:
            pattern_count += 1
            if (pattern_count % 20 == 0):
                tag = "HOLDOUT_3_INPUT" 
            else:
                tag = "MULTI_3_INPUT"

            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        inner = op_eval(op2, b, 1 - c)   # NOT(c)
                        val   = op_eval(op1, a, inner)
                        examples.append((f"{a} {op1} ({b} {op2} NOT({c})) =", str(val), tag))
    return examples


def multiple_operations_6_input():
    """Generate multi-step boolean expressions with 4 different operators."""
    examples = []

    # operator surface forms (same semantics, different text)
    AND_FORMS  = ["AND"]
    OR_FORMS   = ["OR"]
    XOR_FORMS  = ["XOR"]
    NOT_FORMS  = ["NOT"]   # unary
    NAND_FORMS = ["NAND"]       # keep word form (could add "!AND" etc later)
    NOR_FORMS  = ["NOR"]        # keep word form

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):

                            # precompute base ops (ints 0/1)
                            ab_and  = (a and b)
                            ab_or   = (a or b)
                            ab_xor  = (a ^ b)
                            cd_and  = (c and d)
                            cd_or   = (c or d)
                            cd_xor  = (c ^ d)

                            ab_nand = (not (a and b))
                            ab_nor  = (not (a or b))
                            cd_nand = (not (c and d))
                            cd_nor  = (not (c or d))
                            ef_and  = (e and f)
                            ef_or   = (e or f)
                            ef_xor  = (e ^ f)
                            ef_nand = (not (e and f))
                            ef_nor  = (not (e or f))
                            
                            ab_xnor = not (a ^ b)
                            cd_xnor = not (c ^ d)
                            ef_xnor = not (e ^ f)


                            # 1) ((A AND B) AND (C NAND D)) AND (E OR F)
                            for AND in AND_FORMS:
                                for OR in OR_FORMS:
                                    for NAND in NAND_FORMS:
                                        val = ab_and & cd_nand & ef_or
                                        examples.append((
                                            f"(({a} {AND} {b}) {AND} ({c} {NAND} {d})) {AND} ({e} {OR} {f}) =",
                                            str(int(val)),
                                            "MULTI_6_INPUT"
                                        ))

                            # 2) ((A OR B) XOR (C NOR D)) AND (E NAND F)
                            for OR in OR_FORMS:
                                for XOR in XOR_FORMS:
                                    for NOR in NOR_FORMS:
                                        for AND in AND_FORMS:
                                            for NAND in NAND_FORMS:
                                                val = (ab_or ^ cd_nor) & ef_nand
                                                examples.append((
                                                    f"(({a} {OR} {b}) {XOR} ({c} {NOR} {d})) {AND} ({e} {NAND} {f}) =",
                                                    str(int(val)),
                                                    "MULTI_6_INPUT"
                                                ))

                            # 3) NOT(A XOR B) OR ((C AND D) XOR (E OR F))
                            for NOT in NOT_FORMS:
                                for XOR in XOR_FORMS:
                                    for OR in OR_FORMS:
                                        for AND in AND_FORMS:
                                            val = (1 - ab_xor) | (cd_and ^ ef_or)
                                            examples.append((
                                                f"{NOT}({a} {XOR} {b}) {OR} (({c} {AND} {d}) {XOR} ({e} {OR} {f})) =",
                                                str(int(val)),
                                                "MULTI_6_INPUT"
                                            ))

                            # 4) (A NAND B) XOR ((C OR D) AND NOT(E OR F))
                            for NAND in NAND_FORMS:
                                for XOR in XOR_FORMS:
                                    for OR in OR_FORMS:
                                        for AND in AND_FORMS:
                                            for NOT in NOT_FORMS:
                                                val = ab_nand ^ (cd_or & (1 - ef_or))
                                                examples.append((
                                                    f"({a} {NAND} {b}) {XOR} (({c} {OR} {d}) {AND} {NOT}({e} {OR} {f})) =",
                                                    str(int(val)),
                                                    "MULTI_6_INPUT"
                                                ))

                            # 5) (A NOR B) OR ((C XOR D) AND (E AND F))
                            for NOR in NOR_FORMS:
                                for OR in OR_FORMS:
                                    for XOR in XOR_FORMS:
                                        for AND in AND_FORMS:
                                            val = ab_nor | (cd_xor & ef_and)
                                            examples.append((
                                                f"({a} {NOR} {b}) {OR} (({c} {XOR} {d}) {AND} ({e} {AND} {f})) =",
                                                str(int(val)),
                                                "MULTI_6_INPUT"
                                            ))
                                            
                            # ((A AND B) OR (C AND D)) XOR (E OR F)
                            val = ((ab_and) or (cd_and)) ^ (ef_or)
                            examples.append((f"(({a} AND {b}) OR ({c} AND {d})) XOR ({e} OR {f}) =", str(int(val)), "MULTI_6_INPUT"))

                            # ((A XOR B) OR (C XOR D)) AND (E XNOR F)
                            val = ((ab_xor) or (cd_xor)) and (ef_xnor)
                            examples.append((f"(({a} XOR {b}) OR ({c} XOR {d})) AND ({e} XNOR {f}) =", str(int(val)), "MULTI_6_INPUT"))

                            # (A NAND B) OR ((C OR D) AND (E NOR F))
                            val = (ab_nand) or ((cd_or) and (ef_nor))
                            examples.append((f"({a} NAND {b}) OR (({c} OR {d}) AND ({e} NOR {f})) =", str(int(val)), "MULTI_6_INPUT"))

                            # NOT((A OR B) AND (C OR D)) OR (E AND F)
                            val = (not ((ab_or) and (cd_or))) or (ef_and)
                            examples.append((f"NOT(({a} OR {b}) AND ({c} OR {d})) OR ({e} AND {f}) =", str(int(val)), "MULTI_6_INPUT"))

                            # ((A XNOR B) AND (C XOR D)) AND NOT(E OR F)
                            val = (ab_xnor and cd_xor) and (not ef_or)
                            examples.append((f"(({a} XNOR {b}) AND ({c} XOR {d})) AND NOT({e} OR {f}) =", str(int(val)), "MULTI_6_INPUT"))

    return examples
    
    
## unseen dataset builder for testing - patterns should be unique to training set

def multi_operations_4_input_unseen():
    examples = []
    OPS = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]

    

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for op1 in OPS:
                        for op2 in OPS:
                            for op3 in OPS:
                                #((a op1 b) op2 c) op3 d
                                #v = op_eval(op3, op_eval(op2, op_eval(op1, a, b), c), d)
                                #examples.append((f"(({a} {op1} {b}) {op2} {c}) {op3} {d} =", str(v), "UNSEEN_4"))

                                #a op1 (b op2 (c op3 d))
                                #v = op_eval(op1, a, op_eval(op2, b, op_eval(op3, c, d)))
                                #examples.append((f"{a} {op1} ({b} {op2} ({c} {op3} {d})) =", str(v), "UNSEEN_4"))

                                #((a op1 b) op2 c) op3 NOT(d)
                                v = op_eval(op3, op_eval(op2, op_eval(op1, a, b), c), 1 - d)
                                examples.append((f"(({a} {op1} {b}) {op2} {c}) {op3} NOT({d}) =", str(v), "UNSEEN_4"))

                                #NOT(a op1 b) op2 (c op3 d)
                                v = op_eval(op2, 1 - op_eval(op1, a, b), op_eval(op3, c, d))
                                examples.append((f"NOT({a} {op1} {b}) {op2} ({c} {op3} {d}) =", str(v), "UNSEEN_4"))
                                
                                #NOT(a op1 b) op2 NOT(c op3 d)
                                v = op_eval(op2, 1 - op_eval(op1, a, b), 1 - op_eval(op3, c, d))
                                examples.append((f"NOT({a} {op1} {b}) {op2} NOT({c} {op3} {d}) =", str(v), "UNSEEN_4"))
                                
                                

    return examples
    
    
def multi_operations_6_input_unseen():
    """Unseen 6-input templates for robustness testing."""
    examples = []

    AND_FORMS  = ["AND"]
    OR_FORMS   = ["OR"]
    XOR_FORMS  = ["XOR"]
    NOT_FORMS  = ["NOT"]
    NAND_FORMS = ["NAND"]
    NOR_FORMS  = ["NOR"]
    XNOR_FORMS = ["XNOR"]

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):

                            #((A OR C) NAND (B XOR D)) XNOR (E AND F)
                            for OR in OR_FORMS:
                                for NAND in NAND_FORMS:
                                    for XOR in XOR_FORMS:
                                        for XNOR in XNOR_FORMS:
                                            for AND in AND_FORMS:
                                                left  = 1 - ((a | c) & (b ^ d))
                                                right = (e & f)
                                                val = 1 - (left ^ right)
                                                examples.append((
                                                    f"(({a} {OR} {c}) {NAND} ({b} {XOR} {d})) {XNOR} ({e} {AND} {f}) =",
                                                    str(int(val)),
                                                    "UNSEEN_6"
                                                ))

                            #NOT((A NAND F) OR ((B AND E) XOR (C OR D)))
                            for NOT in NOT_FORMS:
                                for NAND in NAND_FORMS:
                                    for OR in OR_FORMS:
                                        for AND in AND_FORMS:
                                            for XOR in XOR_FORMS:
                                                val = 1 - ((1 - (a & f)) | ((b & e) ^ (c | d)))
                                                examples.append((
                                                    f"{NOT}(({a} {NAND} {f}) {OR} (({b} {AND} {e}) {XOR} ({c} {OR} {d}))) =",
                                                    str(int(val)),
                                                    "UNSEEN_6"
                                                ))

                            #((A XNOR B) AND (C NOR D)) OR (E XOR NOT(F))
                            for XNOR in XNOR_FORMS:
                                for AND in AND_FORMS:
                                    for NOR in NOR_FORMS:
                                        for OR in OR_FORMS:
                                            for XOR in XOR_FORMS:
                                                for NOT in NOT_FORMS:
                                                    left = (1 - (a ^ b)) & (1 - (c | d))
                                                    right = e ^ (1 - f)
                                                    val = left | right
                                                    examples.append((
                                                        f"(({a} {XNOR} {b}) {AND} ({c} {NOR} {d})) {OR} ({e} {XOR} {NOT}({f})) =",
                                                        str(int(val)),
                                                        "UNSEEN_6"
                                                    ))

    return examples
  

              


def no_duplicates(examples):
    seen = set()
    out = []
    for prompt, answer, op in examples:
        key = (prompt, op)
        if key in seen:
            continue
        seen.add(key)
        out.append((prompt, answer, op))
    return out

def no_duplicates_prompt_only(examples):
    seen = set()
    out = []
    for prompt, answer, op in examples:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, answer, op))
    return out
            


def build_dataset(test_frac):
    pools = {}

    if AND_OR:
        andor = generate_and_or()
        pools["AND"] = no_duplicates([ex for ex in andor if ex[2] == "AND"])
        pools["OR"] = no_duplicates([ex for ex in andor if ex[2] == "OR"])

    if NAND_NOR:
        nornand = generate_nand_nor()
        pools["NOR"] = no_duplicates([ex for ex in nornand if ex[2] == "NOR"])
        pools["NAND"] = no_duplicates([ex for ex in nornand if ex[2] == "NAND"])
    if XOR_XNOR:
        xornxor = generate_xor_xnor()
        pools["XOR"] = no_duplicates([ex for ex in xornxor if ex[2] == "XOR"])
        pools["XNOR"] = no_duplicates([ex for ex in xornxor if ex[2] == "XNOR"])
        #multi6 = multiple_operations_6_input()
        #pools["MULTI_6_INPUT"] = no_duplicates([ex for ex in multi6 if ex[2] == "MULTI_6_INPUT"])
    if INVERSE:
        inverse = generate_inverse_operations()
        pools["NOT"] = no_duplicates(pools.get("NOT", []) + [ex for ex in inverse if ex[2] == "NOT"])
        #pools["NOT"]  = no_duplicates(pools.get("NOT",  []) + [ex for ex in inverse if ex[2] == "NOT"])

    # robustness-only test patterns (NEW)
    if MULTI_PROMPTS:
        multi_rows = []

        #new generators for 4 inputs in different strictures
        multi_rows += four_input_ops_random_balanced_tree()
        multi_rows += four_input_ops_random_unbalanced_tree()
        #could end up in unseen or holdout or multi depending on toggles
        multi_rows += four_input_ops_random_right_unbalanced_tree_with_not()
        multi_rows += four_input_ops_mixed_bracketing()
        multi_rows += four_input_ops_mixed_with_not_leaves()   
        
        multi_rows += three_input_ops_left_branch()
        multi_rows += three_input_ops_right_branch_with_not() 
        
        multi_rows += for_input_ops_nested_not()
        #multi_rows += for_input_ops_nested_not()
        
        #depending on toggles this will end up in unseen or holdout or multi
        multi_rows += four_input_ops_right_then_left_tree()
        
        
        #unseen patterns for robustness testing
        multi_rows += multi_operations_4_input_unseen()

        #split by tag computed during generation
        pools["MULTI_4_INPUT"] = no_duplicates([ex for ex in multi_rows if ex[2] == "MULTI_4_INPUT"])
        pools["HOLDOUT_4_INPUT"] = no_duplicates([ex for ex in multi_rows if ex[2] == "HOLDOUT_4_INPUT"])
        pools["UNSEEN_4"] = no_duplicates_prompt_only([ex for ex in multi_rows if ex[2] == "UNSEEN_4"])
        
        pools["MULTI_3_INPUT"]   = no_duplicates([ex for ex in multi_rows if ex[2] == "MULTI_3_INPUT"])
        pools["HOLDOUT_3_INPUT"] = no_duplicates([ex for ex in multi_rows if ex[2] == "HOLDOUT_3_INPUT"])

    if SIX_INPUT:
        multi6 = multiple_operations_6_input()
        pools["MULTI_6_INPUT"] = no_duplicates([ex for ex in multi6 if ex[2] == "MULTI_6_INPUT"])
        
        unseen6 = multi_operations_6_input_unseen()
        pools["UNSEEN_6"] = no_duplicates_prompt_only([ex for ex in unseen6 if ex[2] == "UNSEEN_6"])
        
        
        
        

        
    print("Dataset pools sizes:")
    for op, pool in pools.items():
        print(f"  {op}: {len(pool)} examples")

    train_rows = []
    test_rows = []
    stress_test_rows = []

    BASIC_OPS = {"AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"}

    for op, pool in pools.items():
        random.shuffle(pool)

        if op in BASIC_OPS:
            # all basic ops go to training
            #if multi prompts is not on "false" we want basic ops in test set too
            #instead have all basic ops in train and have duplicates in test set too
            train_rows.extend(pool)
            if MULTI_PROMPTS == False:
                #also include in test set dups
                test_rows.extend(pool)
                continue
            continue

        #hard holdout-never train
        if op == "HOLDOUT_4_INPUT":
            test_rows.extend(pool)
            continue
        if op == "UNSEEN_4":
            if UNSEEN_4_IN_TEST: #if enabled put in test set
                test_rows.extend(pool)
                continue
            stress_test_rows.extend(pool)
            continue
        if op == "HOLDOUT_3_INPUT":
            test_rows.extend(pool)
            continue
        if op == "UNSEEN_6":
            #go to test for now but change to stress test later 
            if SIX_TO_TEST: #if enabled put in test set
                test_rows.extend(pool)
                continue
            #if not enabled throw into stress test as an added extra
            stress_test_rows.extend(pool)
            continue
        if op == "MULTI_6_INPUT":
            #these go to train for now
            train_rows.extend(pool)
            continue
        


        n_test = max(1, int(len(pool) * test_frac))
        test_rows.extend(pool[:n_test])
        train_rows.extend(pool[n_test:])

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    train_lines = [f"{p}{a}\n" for (p, a, op) in train_rows]
    
    train_prompts = set(p for (p, a, op) in train_rows)
    test_prompts  = set(p for (p, a, op) in test_rows)
    if MULTI_PROMPTS:
        overlap = train_prompts & test_prompts
        assert len(overlap) == 0, f"Leakage: {len(overlap)} prompts are in both train and test"
    
    return train_lines, test_rows, stress_test_rows


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    train_lines, test_rows, stress_test_rows = build_dataset(TEST_SET)
    
        
    train_path = os.path.join(OUT_DIR,f"bool_{OP}_train.txt")
    test_path = os.path.join(OUT_DIR,f"bool_{OP}_test.tsv")
    stress_test_path = os.path.join(OUT_DIR,f"bool_{OP}_stress_test_unseen.tsv")

    #write training file
    with open(train_path,"w",encoding="utf-8") as f:
        f.writelines(train_lines)
        
    #write tsv prompt/answer/operation test file
    with open(test_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["prompt","answer","operation"])
        for prompt, answer, operation in test_rows:
            writer.writerow([prompt, answer, operation])
            
    #write for stress test unseen patterns
    with open(stress_test_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["prompt","answer","operation"])
        for prompt, answer, operation in stress_test_rows:
            writer.writerow([prompt, answer, operation])
        
    print(f"Wrote {len(train_lines)} training equations to {train_path}")
    print(f"Wrote {len(test_rows)}  test equations to {test_path}")

    print(f"Ops enabled: {OP}")   
    
    
if __name__ == "__main__":
    main()
                 