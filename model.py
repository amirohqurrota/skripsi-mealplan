import csv
import pandas as pd
import numpy as np
import random

def calculatingBody(weight, height):
    bmi = round(float(int(weight)/float(int(height)/100)**2),1)
    if (bmi<18.5):
        bodyLevel = "underweight"
    elif (18.5 < bmi < 24.9):
        bodyLevel = "normal"
    elif (25.0 < bmi < 29.9):
        bodyLevel = "overweight"
    elif (bmi > 30):
        bodyLevel = "obese"
    return bmi, bodyLevel

def calculateNeeds(age,weight,height,gender,activityLevel):
    if (gender=="female"):
            constanta = -161
    else:
        constanta = 5
        
    caloriesDaily = (10* int(weight)) + (6.25*int(height)) - (5*int(age)) + constanta
    if (activityLevel=="sedentary"):
        caloriesDaily = caloriesDaily*1.2
    elif(activityLevel=="lightly"):
        caloriesDaily = caloriesDaily*1.375
    elif(activityLevel=="moderately"):
        caloriesDaily = caloriesDaily*1.55
    elif(activityLevel=="very active"):
        caloriesDaily = caloriesDaily*1.725
    elif(activityLevel=="extra active"):
        caloriesDaily = caloriesDaily*1.9
        
    carboNeeds = 0.6*caloriesDaily/4
    protNeeds = 0.15*caloriesDaily/4
    fatNeeds = 0.15*caloriesDaily/9
    
    bodyLevel = calculatingBody(weight,height)
    
    caloriesDaily=round(caloriesDaily, 1)
    protNeeds=round(protNeeds, 1)
    carboNeeds=round(carboNeeds, 1)
    fatNeeds=round(fatNeeds, 1)

    return bodyLevel,caloriesDaily,protNeeds,carboNeeds,fatNeeds




# =====================generate model section=============

def createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds):
    
    # initiation and read dataframe
    dfFinalEliminated = pd.read_csv("static/data/final_data_done.csv")
    dfFinalEliminated = dfFinalEliminated.dropna(subset=['NAMA BAHAN', 'SUMBER', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)'])
    dfFinalEliminated = dfFinalEliminated.reset_index(drop=True)
    
    def createPopulation(dfFinalEliminated, n_population):
        # pop = np.random.randint(n_meal, size=(n_genes,n_population))
        pop=[]
        for i in range(n_population) :
            sourceCarbs = dfFinalEliminated[dfFinalEliminated['Food Group']=='serealia'].sample(n=3).index
            
            proteinDaging =  dfFinalEliminated[dfFinalEliminated['Food Group']=='daging dan unggas']
            proteinSeafood = dfFinalEliminated[dfFinalEliminated['Food Group']=='prooduk laut']
            proteinDataframe = pd.concat([proteinDaging,proteinSeafood])
            sourceProtAnim = proteinDataframe.sample(n=3).index
            
            sourceVeggies = dfFinalEliminated[dfFinalEliminated['Food Group']=='sayuran'].sample(n=3).index
            sourceSnacks = dfFinalEliminated[dfFinalEliminated['Food Group']=='camilan'].sample(n=1).index
            FruitData = (dfFinalEliminated.loc[(dfFinalEliminated['Food Group']=='buah') & (dfFinalEliminated['Type']=='single')])
            sourceFruit = FruitData.sample(n=1).index
            # pembentukan kromosom
            pop.append([sourceCarbs[0],sourceProtAnim[0],sourceVeggies[0],sourceCarbs[1],sourceProtAnim[1],sourceVeggies[1],sourceCarbs[2],sourceProtAnim[2],sourceVeggies[2],sourceSnacks[0],sourceFruit[0]])
        pop = pd.DataFrame(pop)
        pop.columns = ['breakfast_carbSource','breakfast_protAnimSource','breakfast_veggSource','lunch_carbSource','lunch_protAnimSource','lunch_veggSource','dinner_carbSource','dinner_protAnimSource','dinner_veggSource','snack','fruit']
        return pop

    def randomSelectionParent(n_population):
        position = np.random.permutation(n_population)
        return position[0], position[1]

    def rouletteWheelSelectionParent(pop):
        sizeOfPopulation=pop.shape[0]
        sumOfFitnessPop = sum([chromosome['fitness BaseOn Menu'] for i,chromosome in pop.iterrows()])
        chromosomeProbabilities = [chromosome['fitness BaseOn Menu']/sumOfFitnessPop for i,chromosome in pop.iterrows()]
        arrayIndexPop = [i for i,chromosome in pop.iterrows()]
        choiceParent = np.random.choice(arrayIndexPop,2, p=chromosomeProbabilities)
        return choiceParent[0], choiceParent[1]

    def randomSelectionPosition():
        columns= ['breakfast_carbSource','breakfast_protAnimSource','breakfast_veggSource','lunch_carbSource','lunch_protAnimSource','lunch_veggSource','dinner_carbSource','dinner_protAnimSource','dinner_veggSource','snack','fruit']
        n_pointSize = random.randint(1,3)
        return np.random.choice(columns, size=n_pointSize, replace=False)

    def twopointSelectionPosition():
        n_pointSize = random.randint(1,3)
        position = np.random.choice(range(0,11),2,replace=False)
        position = np.sort(position)
        return position[0], position[1]

    def crossover(pop,n_population):
        # multipoint crossover
        popc = pop.copy()
        for i in range(n_population):
            # inisiasi parent a dan b
            a,b = rouletteWheelSelectionParent(pop)
            c1,c2 = twopointSelectionPosition()
            for i in range(c1,c2) :
                popc.iloc[a][pop.columns[i]], popc.iloc[b][pop.columns[i]] = pop.iloc[b][pop.columns[i]], pop.iloc[a][pop.columns[i]]
        return popc

    def mutation(popc,n_population):
        popm = popc.copy()
        mutationIn = ['breakfast_carbSource','breakfast_protAnimSource','breakfast_veggSource','lunch_carbSource','lunch_protAnimSource','lunch_veggSource','dinner_carbSource','dinner_protAnimSource','dinner_veggSource','snack','fruit']
        source=0
        for i in range(n_population):
            mutation = np.random.choice(mutationIn)
            mutation = str(mutation)
            if (mutation == 'breakfast_carbSource' or mutation == 'lunch_carbSource' or mutation == 'dinner_carbSource'):
                source = dfFinalEliminated[dfFinalEliminated['Food Group']=='serealia'].sample(n=1).index
            elif(mutation == 'breakfast_protAnimSource' or mutation == 'lunch_protAnimSource' or mutation == 'dinner_protAnimSource') :
                source = dfFinalEliminated[dfFinalEliminated['Food Group']=='daging dan unggas'].sample(n=1).index
            elif(mutation == 'breakfast_veggSource' or mutation == 'lunch_veggSource' or mutation == 'dinner_veggSource') :
                source = dfFinalEliminated[dfFinalEliminated['Food Group']=='sayuran'].sample(n=1).index
            elif(mutation=='snack'):
                source = dfFinalEliminated[dfFinalEliminated['Food Group']=='camilan'].sample(n=1).index
            else:
                FruitData = (dfFinalEliminated.loc[(dfFinalEliminated['Food Group']=='buah') & (dfFinalEliminated['Type']=='single')])
                source = FruitData.sample(n=1).index
            popm.iloc[i][mutation]=source[0]
        return popm

    def countFitnessMenu(pop,calNeeds):
        arrayOfPop = pop.to_numpy()
        carboNeeds = 0.6*calNeeds/4
        protNeeds = 0.15*calNeeds/4
        fatNeeds = 0.15*calNeeds/9
        fitnessArray=[]
        caloriesTotalArray=[]
        for chromosome in arrayOfPop:
            index=0
            carboTotal = 0
            protTotal = 0
            fatTotal = 0
            calTotal =0
            for i in (chromosome):
                carbs = dfFinalEliminated.iloc[i]['KH (g)']
                prot = dfFinalEliminated.iloc[i]['PROTEIN (g)']
                fat = dfFinalEliminated.iloc[i]['LEMAK (g)']
                cal = dfFinalEliminated.iloc[i]['ENERGI (Kal)']
                carboTotal+=float(carbs)
                protTotal+=float(prot)
                calTotal+=float(cal)
                fatTotal+=float(fat)
                index+=1
        
            fitness = 1/(abs(carboTotal-carboNeeds)+abs(calTotal-calNeeds)+abs(protNeeds-protTotal)+abs(fatNeeds-fatTotal))
            fitnessArray.append(fitness)
            caloriesTotalArray.append(calTotal)
        
        pop['fitness BaseOn Menu'] = fitnessArray
        pop['calories total Based on Menu'] = caloriesTotalArray
        return pop


    def createPopulationAmount():
        sizeOfPopulation = 5
        sizeOfChromosomes = 11
        pop = np.random.uniform(low = 0.5, high=1.5, size=(sizeOfPopulation,sizeOfChromosomes))
        pop = np.round(pop, decimals=1)
        pop = pd.DataFrame(pop)
        pop.columns = ['nCarbs_br','nProt_br','nVeggies_br','nCarbs_lunch','nProt_lunch','nVeggies_lunch','nCarbs_dinner','nProt_dinner','nVeggies_dinner','n_snack','n_fruit']
        return pop

    def randomSelectionParentAmount(sizeOfPopulation):
        position = np.random.permutation(sizeOfPopulation)
        return position[0], position[1]

    def randomSelectionPositionAmount():
        a=['nCarbs_br','nProt_br','nVeggies_br','nCarbs_lunch','nProt_lunch','nVeggies_lunch','nCarbs_dinner','nProt_dinner','nVeggies_dinner','n_snack','n_fruit']
        return np.random.choice(a, size=1, replace=False)[0]

    def crossoverAmount(pop,sizeOfPopulation):
        popc = pop.copy()
        for i in range(sizeOfPopulation):
        # inisiasi parent a dan b
            a,b = randomSelectionParentAmount(sizeOfPopulation)
            swap = randomSelectionPositionAmount()
            # crossover
            popc.iloc[a][swap], popc.iloc[b][swap] = pop.iloc[b][swap], pop.iloc[a][swap]
        return popc

    def arithmeticCrossoverAmount(pop,sizeOfPopulation):
        alpha=random.uniform(0.1,0.9)
        alpha=round(alpha, 1)
        betha=1-alpha
        betha=round(betha, 1)
        
        popc = pop.copy()
        for i in range (sizeOfPopulation):
            a,b = randomSelectionParentAmount(sizeOfPopulation)
            crossoverResult = []
            for j in range (11):
                value = round(alpha * pop.iloc[a][pop.columns[j]] + betha * pop.iloc[b][pop.columns[j]],1)
                crossoverResult.append(value)
                popc.loc[i , pop.columns[j]]=value
        return popc


    def mutationAmount(pop, sizeOfPopulation):
        for i in range(sizeOfPopulation):
            a,b = twopointSelectionPosition()
            for j in range(a,b) :
                alpha=random.uniform(-0.5,0.4)
                alpha=round(alpha, 1)
                pop.iloc[i][pop.columns[j]]=abs(round(pop.iloc[i][pop.columns[j]] + alpha ,1))
        return pop

    def fitnessAmount(popAmount,chromosomesMenu,calNeeds):
        arrayOfAmount = popAmount.to_numpy()
        carboNeeds = 0.6*calNeeds/4
        protNeeds = 0.15*calNeeds/4
        fatNeeds = 0.15*calNeeds/9
        fitnessArray=[]
        caloriesTotalArray=[]
        for amount in arrayOfAmount:
            index=0
            carboTotal = 0
            protTotal = 0
            fatTotal = 0
            calTotal =0
            for i in range(11):
                carbs = float(dfFinalEliminated.iloc[int(chromosomesMenu[i])]['KH (g)'])*amount[index]
                prot = float(dfFinalEliminated.iloc[int(chromosomesMenu[i])]['PROTEIN (g)'])*amount[index]
                fat = float(dfFinalEliminated.iloc[int(chromosomesMenu[i])]['LEMAK (g)'])*amount[index]
                cal = float(dfFinalEliminated.iloc[int(chromosomesMenu[i])]['ENERGI (Kal)'])*amount[index]
                carboTotal+=carbs
                protTotal+=prot
                calTotal+=cal
                fatTotal+=float(fat)
                index+=1
            fitness = 1/(abs(carboTotal-carboNeeds)+abs(calTotal-calNeeds)+abs(protNeeds-protTotal)+abs(fatNeeds-fatTotal))
            fitnessArray.append(fitness)
            caloriesTotalArray.append(calTotal)
            
        popAmount['fitness'] = fitnessArray
        popAmount['calories total'] = caloriesTotalArray
        return popAmount

    def decideAmountThroughGeneticAlgorithm(chromosomes,calNeeds):
        # Carbohydrates provide 4 calories per gram, protein provides 4 calories per gram, and fat provides 9 calories per gram.
        carboNeeds = 0.6*calNeeds/4
        protNeeds = 0.15*calNeeds/4
        fatNeeds = 0.15*calNeeds/9
        goal_amount={
        'nCarbs_br': [-1], 'nProt_br': [-1],'nVeggies_br': [-1],
        'nCarbs_lunch':[-1],'nProt_lunch':[-1],'nVeggies_lunch':[-1],
        'nCarbs_dinner':[-1],'nProt_dinner':[-1],'nVeggies_dinner':[-1],'nSnack':[-1],'nFruit':[-1],
        'fitness': [-1000000], 'caloriesTot':[0]}
        goal_amount=pd.DataFrame(goal_amount)
        current_amount=pd.DataFrame()
        chromosomes = chromosomes.to_numpy(dtype = int)[0]
        chromosomes = chromosomes[0:11]
        iteration=5
        # for i in range(iteration):
        # isGlobalOptimum=falses
        n=0
        for i in range(iteration):
            pop=createPopulationAmount()
            pop=fitnessAmount(pop,chromosomes,calNeeds)
            # popc=crossoverAmount(pop,5)
            popc=arithmeticCrossoverAmount(pop,5)
            popc=fitnessAmount(popc,chromosomes,calNeeds)
            # popm=mutationAmount(popc,5)
            current_amount = popc.sort_values(by = 'fitness',ascending = True).iloc[[0]]
            if goal_amount.iloc[0]["fitness"]<current_amount.iloc[0]["fitness"]:
                goal_amount.iloc[[0]]=current_amount.iloc[[0]]
            # if (n==10):
            #       isGlobalOptimum=True
            # if ():
            #       n+=1
            # print(current_amount.iloc[0]["fitness"], " ==== ", i_chromosomes)
        return goal_amount

    def viewResultByOptimalAmount(popFinalAmount,chromosomes,calNeeds):
        carboNeeds = 0.6*calNeeds/4
        protNeeds = 0.15*calNeeds/4
        fatNeeds = 0.15*calNeeds/9
        # chromosomes = chromosomes.pop["fitness"]
        amountArray=popFinalAmount.to_numpy()[0]
        chromosomes = chromosomes.to_numpy()[0]
        chromosomes = chromosomes[0:11]
        index=0
        finalCarbs = 0
        finalProt = 0
        finalFat = 0
        finalCal = 0
        for i in (chromosomes):
            carbs = float(dfFinalEliminated.iloc[int(i)]['KH (g)'])*amountArray[index]
            prot = float(dfFinalEliminated.iloc[int(i)]['PROTEIN (g)'])*amountArray[index]
            fat = float(dfFinalEliminated.iloc[int(i)]['LEMAK (g)'])*amountArray[index]
            cal = float(dfFinalEliminated.iloc[int(i)]['ENERGI (Kal)'])*amountArray[index]
            finalCarbs += carbs
            finalProt += prot
            finalFat += fat
            finalCal += cal
            index+=1
        print ("kebutuhan kalori - real kalori : " ,calNeeds , " - ", finalCal)
        print ("kebutuhan karbo - real karbo : " ,carboNeeds , " - ", finalCarbs)
        print ("kebutuhan prot - real prot : " ,protNeeds , " - ", finalProt)
        print ("kebutuhan lemak - real lemak : " ,fatNeeds , " - ", finalFat)
        print("\n")
        print ("===BREAKFAST===")
        print("sumber karbohidrat : " ,dfFinalEliminated.iloc[int(chromosomes[0])]['NAMA BAHAN'], " ",100*amountArray[0], "gr")
        print("sumber protein : " , dfFinalEliminated.iloc[int(chromosomes[1])]['NAMA BAHAN'], " ",100*amountArray[1], "gr")
        print("sumber sayuran/protein nabati : " , dfFinalEliminated.iloc[int(chromosomes[2])]['NAMA BAHAN'], " ",100*amountArray[2], "gr")
        print ("===LUNCH===")
        print("sumber karbohidrat : " ,dfFinalEliminated.iloc[int(chromosomes[3])]['NAMA BAHAN'], " ",100*amountArray[3], "gr")
        print("sumber protein : " , dfFinalEliminated.iloc[int(chromosomes[4])]['NAMA BAHAN'], " ",100*amountArray[4], "gr")
        print("sumber sayuran/protein nabati : " , dfFinalEliminated.iloc[int(chromosomes[5])]['NAMA BAHAN'], " ",100*amountArray[5], "gr")
        print ("===DINNER===")
        print("sumber karbohidrat : " ,dfFinalEliminated.iloc[int(chromosomes[6])]['NAMA BAHAN'], " ",100*amountArray[6], "gr")
        print("sumber protein : " , dfFinalEliminated.iloc[int(chromosomes[7])]['NAMA BAHAN'], " ",100*amountArray[7], "gr")
        print("sumber sayuran/protein nabati : " , dfFinalEliminated.iloc[int(chromosomes[8])]['NAMA BAHAN'], " ",100*amountArray[8], "gr")
        print("===CAMILAN===")
        print("camilan pagi (buah) : " , dfFinalEliminated.iloc[int(chromosomes[9])]['NAMA BAHAN'], " ",100*amountArray[9], "gr")
        print("camilan sore : " , dfFinalEliminated.iloc[int(chromosomes[10])]['NAMA BAHAN'], " ",100*amountArray[10], "gr")
    n_iteration=5
    arrayOfIteration = []
    arrayOfFitness = []
    defCal=False
    # popAmount=[]
    popAmount={
      'nCarbs_br': [], 'nProt_br': [],'nVeggies_br': [],
      'nCarbs_lunch':[],'nProt_lunch':[],'nVeggies_lunch':[],
      'nCarbs_dinner':[],'nProt_dinner':[],'nVeggies_dinner':[],'nSnack':[],'nFruit':[],
      'fitness': [], 'caloriesTot':[]
      }
    goal_menu={
      'sourceCarbs_br': [-1], 'sourceProt_br': [-1],'sourceVeggies_br': [-1],
      'sourceCarbs_lunch':[-1],'sourceProt_lunch':[-1],'sourceVeggies_lunch':[-1],
      'sourceCarbs_dinner':[-1],'sourceProt_dinner':[-1],'sourceVeggies_dinner':[-1],'sourceSnack':[-1],'sourceFruit':[-1],
      'fitness optimal': [-1000000]
      }
    goal_amount={
      'nCarbs_br': [-1], 'nProt_br': [-1],'nVeggies_br': [-1],
      'nCarbs_lunch':[-1],'nProt_lunch':[-1],'nVeggies_lunch':[-1],
      'nCarbs_dinner':[-1],'nProt_dinner':[-1],'nVeggies_dinner':[-1],'nSnack':[-1],'nFruit':[-1],
      'fitness': [-1000000], 'caloriesTot':[0]
      }
    goal_menu=pd.DataFrame(goal_menu)
    goal_amount=pd.DataFrame(goal_amount)
    popAmount=pd.DataFrame(popAmount)
    for i in range (n_iteration):
        fitnessArray=[]
        menuPop = createPopulation(dfFinalEliminated,5)
        menuPop = countFitnessMenu(menuPop,calNeeds)
        menuPopC=crossover(menuPop,5)
        menuPopM=mutation(menuPop,5)
        for j in range(len(menuPopM)):
            bestAmount = decideAmountThroughGeneticAlgorithm(menuPopM.iloc[[j]],calNeeds)
            # input final amount to amount dataframe to save best amount in current menu
            popAmount = pd.concat([popAmount,bestAmount.iloc[[0]]], ignore_index = True)
              # add fitness to dataframe menu
            fitnessArray.append(bestAmount._get_value(0,'fitness'))
            # menuPopM.iloc[i]['fitness']= bestAmount._get_value(0,'fitness')
        
        menuPopM['fitness optimal'] = fitnessArray   
        conditionArray=np.array([])
        for k in range (len(popAmount)):
            if (defCal):
                condition=True
                if (popAmount.iloc[k]['caloriesTot']>calNeeds):
                    # takeout chromosomes
                    condition=False
                np.append(conditionArray,condition)
            else :
                condition=True
                if (popAmount.iloc[k]['caloriesTot']<calNeeds):
                    # takeout chromosomes
                    condition=False
                np.append(conditionArray,condition)
        getIndexOfCondition=np.where(conditionArray==False)[0]
        menuPopM.drop(getIndexOfCondition)
        popAmount.drop(getIndexOfCondition)

        
        if goal_menu.iloc[0]['fitness optimal'] < menuPopM.sort_values(by = 'fitness optimal',ascending = False).iloc[0]['fitness optimal']:
              goal_menu = menuPopM.sort_values(by = 'fitness optimal',ascending = False).iloc[[0]]
              goal_amount = popAmount.sort_values(by = 'fitness',ascending = False).iloc[[0]]
        
        arrayOfFitness.append(goal_menu.iloc[0]['fitness optimal'])
        arrayOfIteration.append(i)
        
    class MealPlan:
        def __init__(self,dfMeal,dfAmount):
            amount=dfAmount.to_numpy()[0]
            amount = amount[0:11]
            meal = dfMeal.to_numpy()[0]
            meal = meal[0:11]
            # arrayNameMeal = ["meal_breakfast_carb","meal_breakfast_prot","meal_breakfast_veg","meal_lunch_carb","meal_lunch_prot","meal_lunch_veg","meal_dinner_carb","meal_dinner_prot","meal_dinner_veg","meal_breakfast_snack","meal_dinner_snack",]
            # arrayAmountMeal = ["amount_breakfast_carb","amount_breakfast_prot","amount_breakfast_veg","amount_lunch_carb","amount_lunch_prot","amount_lunch_veg","amount_dinner_carb","amount_dinner_prot","amount_dinner_veg","amount_breakfast_snack","amount_dinner_snack",]
            self.meal_breakfast_carb = dfFinalEliminated.iloc[int(meal[0])]['NAMA BAHAN']
            self.meal_breakfast_prot = dfFinalEliminated.iloc[int(meal[1])]['NAMA BAHAN']
            self.meal_breakfast_veg = dfFinalEliminated.iloc[int(meal[2])]['NAMA BAHAN']
            self.meal_lunch_carb =dfFinalEliminated.iloc[int(meal[3])]['NAMA BAHAN']
            self.meal_lunch_prot =dfFinalEliminated.iloc[int(meal[4])]['NAMA BAHAN']
            self.meal_lunch_veg = dfFinalEliminated.iloc[int(meal[5])]['NAMA BAHAN']
            self.meal_dinner_carb = dfFinalEliminated.iloc[int(meal[6])]['NAMA BAHAN']
            self.meal_dinner_prot = dfFinalEliminated.iloc[int(meal[7])]['NAMA BAHAN']
            self.meal_dinner_veg = dfFinalEliminated.iloc[int(meal[8])]['NAMA BAHAN']
            self.meal_breakfast_snack = dfFinalEliminated.iloc[int(meal[9])]['NAMA BAHAN']
            self.meal_dinner_snack = dfFinalEliminated.iloc[int(meal[10])]['NAMA BAHAN']
            # for i in range (len(meal)):
            #     name = arrayNameMeal[i]
            #     self.name=dfFinalEliminated.iloc[[meal[i]]]
                
            # for j in range (len(amount)):
            #     name = arrayAmountMeal[j]
            #     self.name=amount[j]*100
            
            self.amount_breakfast_carb = amount[0]
            self.amount_breakfast_prot = amount[1]
            self.amount_breakfast_veg = amount[2]
            self.amount_lunch_carb = amount[3]
            self.amount_lunch_prot = amount[4]
            self.amount_lunch_veg = amount[5]
            self.amount_dinner_carb = amount[6]
            self.amount_dinner_prot = amount[7]
            self.amount_dinner_veg = amount[8]
            self.amount_breakfast_snack = amount[9]
            self.amount_dinner_snack = amount[10]
            
            index=0
            finalCarbs = 0
            finalProt = 0
            finalFat = 0
            finalCal = 0
            for i in (meal):
                carbs = float(dfFinalEliminated.iloc[int(i)]['KH (g)'])*amount[index]
                prot = float(dfFinalEliminated.iloc[int(i)]['PROTEIN (g)'])*amount[index]
                fat = float(dfFinalEliminated.iloc[int(i)]['LEMAK (g)'])*amount[index]
                cal = float(dfFinalEliminated.iloc[int(i)]['ENERGI (Kal)'])*amount[index]
                finalCarbs += carbs
                finalProt += prot
                finalFat += fat
                finalCal += cal
                index+=1
            
            self.total_carbs = finalCarbs
            self.total_prot = finalProt
            self.total_fat = finalFat
            self.total_cal = finalCal

        
    mealDay1= MealPlan(goal_menu.iloc[[0]],goal_amount.iloc[[0]])
    
    
    
    # viewResultByOptimalAmount(goal_amount.iloc[[0]],goal_menu.iloc[[0]],calNeeds)
    # return arrayOfFitness,arrayOfIteration
    
    
    # return goal_amount.iloc[[0]],goal_menu.iloc[[0]]
    return mealDay1
    


def create7DaysMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds):
    Meal1 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal2 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal3 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal4 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal5 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal6 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    Meal7 = createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
    # totalCarbsInADay = Meal1.total_carbs + 
    return Meal1,Meal2,Meal3,Meal4,Meal5,Meal6,Meal7

# meal1,meal2,meal3,meal4,meal5,meal6,meal7=create7DaysMealPlan(1900,73,291,32)
# listMeal = [meal1,meal2,meal3,meal4,meal5,meal6,meal7]
# # print(listMeal[1])
# for i in (listMeal):
#     print("========")
#     attrs = vars(i)
#     print(', '.join("%s: %s" % item for item in attrs.items()))

