#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#-------		q_calculator.py library			- Written by Liam Booth---------#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# REV 1.0 Initial release, screen manager added - 11/12/2020 10:30am			#
# REV 1.1 Data plotter added - 03/03/2020 14:22pm								#
# REV 1.2 Q calculator added - 04/03/2020 15:17pm								#
# REV 1.3 Added Q functions - 15/04/2020 11:11am								#
#-------------------------------------------------------------------------------#
#++++++++++++++++++++++++++ Notes and comments! ++++++++++++++++++++++++++++++++#
# Has many functions for creating andsorting through pregenerated sqlite 		#
# libraries, also calculating q value based off Thomas(1963).					#
#-------------------------------------------------------------------------------#
# Any issues or comments please contact - 	liam.booth@hull.ac.uk				#
#-------------------------------------------------------------------------------#
from math import log10
import random
# Calculates q value according too other papers and Thomas, 3(x+y) for carry over and 3(x+y)+2 for not a carry over
def Q_addition_old(int1, int2):
	Q, c, offset = 0, 0, 1
	if int1 == 0 or int2 == 0: return 0
	while (offset-1 < max(len(str(int1)),len(str(int2)))): # iterate through while loop, creating a summation of q value until no digits are left
		# get last integer of random number
		if(offset>len(str(int1))):	x = 0
		else:						x = int(str(int1)[-offset])
		if(offset>len(str(int2))):	y = 0
		else: 						y = int(str(int2)[-offset])
		print(x+y)
		if x == 0 or y == 0:
			total = 1
		else:	
			if x+y >= 10: # make sure added digits are greater then 10 for carry
				print("rule 1")
				total = (3*(x+y))+c # subsitute for x+y+(x+y)+c+(x+y+c), is added if cary from previous occurs
				c = 2 
			elif 	(x+y <= 10) and  (x+y > 0):	# if not 0 or carry over
				print("rule 2")
				total = (3*(x+y))+c # subsitute for x+y+(x+y)+c+(x+y+c)
				c = 0
			else: # If single digit and no carry
				print("rule 3")
				if x+y == 0:
					total = 1
				else:
					total = 2*(x+y) # subsitute for x+y+(x+y)
					c = 0
		Q = Q + log10(total) # add total log to find over Q value  
		offset = offset + 1
	return Q
	
# Calculates q value for liams extended rules for Thomas, 3(x+y) for carry over and 3(x+y)+2 for not a carry over
def Q_addition_long(int1, int2):
	Q = 0 		# Q = Totsl q-value, 
	c = 0 		# c = Carry over occured from previous position, 
	offset = 1 	# offset = Tracker for current digit pair
	maxint = max(len(str(int1)), len(str(int2))) # mind number with largest maount of digits
	total = 1
	if int1 == 0 or int2 == 0: return 0 # catch if either whole number is 0, return Q = 0
	while (offset-1 < maxint): # iterate through while loop, creating a summation of q value until no digits are left
		# get next integer for each number
		if(offset>len(str(int1))):	x = 0
		else:						x = int(str(int1)[-offset])
		if(offset>len(str(int2))):	y = 0
		else: 						y = int(str(int2)[-offset])
		#print(x+y)
		if x == 0 or y == 0: # if either digit is 0 set total to 1, (log of 1 is 0)
			total = 1
		else:	
			if (x+y >= 10) and (c == 2) and (offset < maxint):
				#print("rule 1")
				total = (4*(x+y))+c
				c = 2 
			elif (x+y >= 10) and  (c == 0) and (offset < maxint) :
				#print("rule 2")
				total = (3*(x+y))
				c = 2	
			elif (x+y >= 10) and  (c == 0) and (maxint == 1) :
				#print("rule 3")
				total = (2*(x+y))
				c = 2		
			elif (x+y < 10) and  (c == 0):	
				#print("rule 4")
				total = (2*(x+y))
				c = 0
			elif (x+y < 10) and  (c == 2 ) and (offset < maxint):
				#print("rule 5")
				total = (3*(x+y)+c)
				c = 0
			elif (x+y >= 10) and  (c == 2 ) and (offset == maxint):
				#print("rule 6")
				total = (3*(x+y)+c) 
				c = 0	
		Q = Q + log10(total) # add total log to find over Q value  
		offset = offset + 1
	return Q	

# Calculates q value according too other papers and Thomas, 3(x+y) for carry over and 3(x+y)+2 for not a carry over
def Q_addition(int1, int2):
	Q = 0 		# Q = Totsl q-value, 
	c = 0 		# c = Carry over occured from previous position, 
	offset = 1 	# offset = Tracker for current digit pair
	maxint = max(len(str(int1)), len(str(int2))) # mind number with largest maount of digits
	if int1 == 0 or int2 == 0: return 0 # catch if either whole number is 0, return Q = 0
	while (offset-1 < maxint): # iterate through while loop, creating a summation of q value until no digits are left
		# get next integer for each number, if no number set to 0
		if(offset>len(str(int1))):	x = 0
		else:						x = int(str(int1)[-offset])
		if(offset>len(str(int2))):	y = 0
		else: 						y = int(str(int2)[-offset])
		factor = 0
		if x == 0 or y == 0:		pass # if either digit is 0 set total to 1, (log of 1 is 0)
		else:	
			if c == 2: 				factor = 1# if carry from previous increase factor
			if (x+y >= 10) and (offset < maxint):
				Q += log10((factor+3)*(x+y)+c)
				c = 2
			else:	
				Q += log10((factor+2)*(x+y)+c)
				c = 0	
		offset = offset + 1 # shift to next digit pair
	return Q # return total Q
	
def find_elements(int1,int2):
	elements = len(str(int1))+len(str(int2))
	totallog = 0
	total = 1
	int1len = len(str(int1))
	int2len = len(str(int2))
	if not int1 and not int2:
		totallog = 0
	# Click link to find related article where it describes q value for integer arithmetic difficulty at https://link.springer.com/content/pdf/10.1007%2Fs11858-015-0754-8.pdf
	while int1 or int2: # iterate through while loop, creating a summation of q value until no digits are left
		# get last integer of random number
		if int1: # check if there is digit, if not set to 0
			x = int(str(int1)[-1:])
		else:
			x = 0
		if int2: # check if there is digit, if not set to 0
			y = int(str(int2)[-1:])
		else:
			y = 0
		if (int1 and int2):
			if (int1 and int2):
				if x+y >= 10 and ((int1len != 1) and (int2len != 1)):
					elements = elements + 1
		if int1: # check if there is digit, if not set to dont attempt character removal
			int1 = str(int1)[:-1] # removes smallest digit from int1
		else: 
			int1 = 0
		if int2: # check if there is digit, if not set to dont attempt character removal
			int2 = str(int2)[:-1] # removes smallest digit from int2
		else: 
			int2 = 0
	return float(elements)
    
def Generate_Q_Question(qmin, qmax, intmax):
    curq = 0 
    if qmin >= qmax:
        return None
    n1 = random.randint(0,intmax)
    n2 = random.randint(0,intmax)
    curq = Q_addition_long(n1, n2)    
    while (curq < qmin) or (curq > qmax):
        n1 = random.randint(0,intmax)
        n2 = random.randint(0,intmax)
        curq = Q_addition_long(n1, n2)  
        #print(n1,n2,curq)
    return [n1,n2]
    