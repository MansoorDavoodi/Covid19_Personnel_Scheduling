#
#   https://gekko.readthedocs.io/en/latest/
#   https://apmonitor.com/wiki/index.php/Main/OptionApmSolver


#   Personnel Scheduling under Covid 19 Pandemic: ONLY presence strategy as the decision variable
#   Modeling and details are available at   https://www.overleaf.com/project/623996c0953e543e97c03fa3
#
#   Two (GEKKO + IPOPT), GA and Gurobi approaches are presented in this code.


import math
import networkx as nx
from collections import namedtuple

import xlwt 
from xlwt import Workbook
import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from random import randint
from gekko import GEKKO

import pyomo.environ as pe
import pyomo.opt as po

########################################################################################
######  IO Section

def convert_Gekko_type_to_data(Presences, Probs):
    n, T = Presences.shape
    WPresences = np.zeros((n, T))
    for i in range(n):
        for j in range(T):
            if Presences[i, j].value[0] > 0.5 :
                WPresences[i, j] = 1
            else:
                WPresences[i, j] = 0
            
    n, T = Probs.shape
    WProbs = np.zeros((n, T))
    for i in range(n):
        for j in range(T):
            WProbs[i, j] = Probs[i, j].value[0]
    
    return WPresences, WProbs

def convert_gurobi_type_to_data(Presences, Probs, n,T):
    WPresences = np.zeros((n, T))
    for i in range(n):
        for j in range(T):
            WPresences[i, j] = pe.value(Presences[i, j])
            if (WPresences[i, j] != 1 and WPresences[i, j] != 0):
                if WPresences[i, j] > 0.5:
                    WPresences[i, j] = 1
                else:
                    WPresences[i, j] = 0
                    
    WProbs = np.zeros((n, T+1))
    for i in range(n):
        for j in range(T+1):
            WProbs[i, j] = pe.value(Probs[i, j])
    
    return WPresences, WProbs

def read_input_graph(file_path):
    df = pd.read_excel(file_path, engine  = 'openpyxl')
    n = df.shape[0]
    Weights = np.zeros((n, n))
    Vaccine = list()
    Group = list()
    for i in range(n):
        Data = df.values[i, :]
        for j in range (n):
            Weights[i, j] = Data[j+1]
        Vaccine.append(Data[n+1])
        Group.append(Data[n+2])

    return Weights, Vaccine, Group

def draw_physical_graph(Weights, Vaccine):
    n, temp = Weights.shape
    
    G = nx.Graph()
   
    Color = namedtuple("Color", "R G B")
    nodecolorg = Color(0/255, 255/255, 0/255) # green
    nodecolorr = Color(255/255, 0/255, 0/255) # red
    
    for i in range (0, n):
        if Vaccine[i]  ==  1:
            G.add_node(str(i+1), color = nodecolorg)
        else:
            G.add_node(str(i+1), color = nodecolorr)
        
    for i in range (0, n):
        for j in range (i, n):
            if Weights[i, j] > 0:
                G.add_edges_from([(str(i+1), str(j+1))], weight = str(Weights[i, j]))
    
    colors = [node[1]['color'] for node in G.nodes(data = True)]
    #nx.draw(G, node_size = 600, node_color = colors, with_labels = True, font_color = 'black')
    
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True, node_size = 400, node_color = colors)
    
    edge_labels = dict([((u, v), d['weight'])
                    for u, v, d in G.edges(data = True)])

    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, label_pos = 0.6, 
                             font_color =  'blue', font_size = 7)
    
    plt.axis('on')
    plt.show()    

def generate_random_graph(n, weightprob, vaccinprob, groupnum):
    Weights = np.zeros((n, n))
    Vaccine = np.zeros(n)
    Group = np.zeros(n)
    
    for i in range(n):
        if (randint(0, 100)/100)  <=  vaccinprob:
            Vaccine[i] = 1
        Group[i] = randint(0, groupnum)
        
        for j in range(i + 1, n):
            x = randint(0, 1000)/1000
            if(x<weightprob):
                Weights[i, j] = 1
            elif(x < 2 * weightprob):
                Weights[i, j] = 0.5
            else:
                Weights[i, j] = 0
                
            Weights[j, i] = Weights[i, j]
            
    
    return Weights, Vaccine, Group

def double_check_Obj_Function(Presences, Probs, 
                          Weights, Vaccine, Group, InitProbs, 
                            time_period, accuracy_of_tests, betau, betav,pt):
    n, T = Probs.shape
    A = np.zeros((n, T))
    AL = np.zeros((n, T))
    FN = 1-accuracy_of_tests
    
    for i in range (n):
        A[i, 0] = InitProbs[i]
        AL[i, 0] = InitProbs[i]
        
    for j in range (T-1):
        for i in range (n):
            trans = betav * Vaccine[i]+betau * (1-Vaccine[i])
            
            temp = 1
            for k in range (n):
                temp = temp *  (1 - Weights[i, k] * trans * Probs[k, j] * Presences[k, j] * (pt * FN + (1-pt)))
            
            temp2 = A[i, j] * pt * FN + A[i, j] * (1-pt)
            temp3 = 1 - ((1-A[i, j]) * temp)
            A[i, j+1] =  temp2 * (1 - Presences[i, j]) + Presences[i, j] * temp3
    
    for j in range (T-1):
        for i in range (n):
            trans = betav * Vaccine[i]+ betau * (1-Vaccine[i])
            
            ## Linearization
            temp = 0
            for k in range (n):
                temp = temp +  (Weights[i, k] * trans * AL[k, j] * Presences[k, j] * (pt * FN + (1-pt)))
            temp = 1 - temp
            
            temp2 = AL[i, j] * pt * FN + AL[i, j] * (1-pt)
            temp3 = 1 - ((1-AL[i, j]) * temp)
            AL[i, j+1] =  temp2 * (1 - Presences[i, j]) + Presences[i, j] * temp3
    
    obj = 0
    obj_l = 0
    cnt = 0
    T = time_period
    for j in range(0, T):
        for i in range(n):
            cnt = cnt + Presences[i, j]
            if j < T - 1:
                obj = obj + A[i, j + 1] * Presences[i, j]
                obj_l = obj_l + AL[i, j + 1] * Presences[i, j]
            else:
                obj = obj + 2 * A[i, j+1] * Presences[i, j]
                obj_l = obj_l + 2 * AL[i, j+1] * Presences[i, j]
            
    obj = obj / (n * (T + 1))
    obj_l = obj_l / (n * (T + 1))
    
    print("double check (linear): ", obj_l)
    print("double check (exact ): ", obj)
    
    return obj_l, obj

def drawing_graph_day(Weights, Presence_day, Probs_day, background_risk, daylabel):
    G = nx.Graph()
    n, temp = Probs.shape
    Color = namedtuple("Color", "R G B")
    
    color1 = Color(0/255, 255/255, 0/255) # green
    color2 = Color(30/255, 220/255, 0/255)
    color3 = Color(50/255, 205/255, 0/255)
    color4 = Color(120/255, 180/255, 0/255)

    color5 = Color(140/255, 140/255, 0/255)
    color6 = Color(200/255, 200/255, 0/255)

    
    color7 = Color(150/255, 100/255, 0/255)
    color8 = Color(170/255, 75/255, 0/255)
    color9 = Color(200/255, 50/255, 0/255)
    color10 = Color(255/255, 0/255, 0/255) #red

    for i in range (0, n):
        if(Probs_day[i]  >=  1.8 * background_risk):
            mycolor = color10
        elif(Probs_day[i]  >=  1.6 * background_risk):
            mycolor = color9
        elif(Probs_day[i]  >=  1.4 * background_risk):
            mycolor = color8
        elif(Probs_day[i]  >=  1.2 * background_risk):
            mycolor = color7
        elif(Probs_day[i]  >=  1.00 * background_risk):
            mycolor = color6
        elif(Probs_day[i]  >=  0.8 * background_risk):
            mycolor = color5
        elif(Probs_day[i]  >=  0.6 * background_risk):
            mycolor = color4
        elif(Probs_day[i]  >=  0.4 * background_risk):
            mycolor = color3
        elif(Probs_day[i]  >=  0.2 * background_risk):
            mycolor = color2
        else:
            mycolor = color1

        #options = {'node_color': 'black', 'node_size': 50, 'width': 2}
        
        if Presence_day[i] == 1:
                G.add_node(str(i+1), color = mycolor)

    for i in range (0, n):
        for j in range (i, n):
            if Presence_day[i] == 1 and Presence_day[j] == 1:
                if Weights[i, j] > 0:
                        G.add_edges_from([(str(i+1), str(j+1))], width = (math.ceil(Weights[i, j] * 3)))

    #plt.figure(figsize = (10, 8))
    ax = plt.gca()
    ax.set_title('Day '+str(daylabel))
    
    colors = [node[1]['color'] for node in G.nodes(data = True)]
    
    edge_widths = [G[i][j]['width'] for (i, j) in G.edges()]
    
    nx.draw(G, node_size = 350, node_color = colors, with_labels = True, font_color = 'black', ax = ax, width = edge_widths)

    _ = ax.axis('on')

    plt.show()

def drawing_graph_week(Weights, Presences, Probs, background_risk):
    n, days = Presences.shape
    for i in range(days):
        Probs_day = Probs[:, i+1]
        Presence_day = Presences[:, i]
        drawing_graph_day(Weights, Presence_day, Probs_day, background_risk, daylabel = i+1)

def report_excel_output(Presences, obj, objlinear, objexact, elapsed_time, out_file_path):
    n = Presences.shape[0]
    T = Presences.shape[1]

    wb = Workbook() 
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('Presence Schedule') 

    style1 = xlwt.easyxf('font: bold 1, color blue;') 
    style2 = xlwt.easyxf('font: color red;') 

    #sheet1.write(0, 2, 'n', style)
    if T == 5:
        sheet1.write(0, 1, 'Monday', style1)
        sheet1.write(0, 2, 'Tuesday', style1)
        sheet1.write(0, 3, 'Wednesday', style1)
        sheet1.write(0, 4, 'Thursday', style1)
        sheet1.write(0, 5, 'Friday', style1)
    else:
        for t in range (T):
            s = 'Day '+str(t+1)
            sheet1.write(0, t+1, s, style1)
    for i in range (n):
        s = 'employee '+str(i+1)
        sheet1.write(i+1, 0, s, style1)
    
    for i in range (n):
        for t in range (T):
            s = Presences[i, t]
            sheet1.write(i+1, t+1, s)
            
    sheet1.write(n+2, 0, "Expected Risk (solver):",style2)
    sheet1.write(n+2, 1, obj,style2)
    sheet1.write(n+3, 0, "Expected Risk (linear):",style2)
    sheet1.write(n+3, 1, objlinear,style2)
    sheet1.write(n+4, 0, "Expected Risk (exact):",style2)
    sheet1.write(n+4, 1, objexact,style2)
    
    sheet1.write(n+6, 0, "Running Time:")
    sheet1.write(n+6, 1, elapsed_time)

    wb.save(out_file_path) 

#####  end of IO Section
######################################################################################
#########  GEKKO 

def Gekko_cons_func(Presences, Probs, trans, Weights, accuracy_of_tests, test_probability_per_day, i, j):
    FN = 1 - accuracy_of_tests
    n, temp = Weights.shape
    pt = test_probability_per_day
    
    ####################
    # temp = 1
    # for k in range (n):
    #     temp = temp *  (1-Weights[i, k] * trans * Probs[k, j] * Presences[k, j] * (pt * FN + (1-pt)))
    #################################
        
    # Linearization
    temp = 0
    for k in range (n):
        #if (Presences[k,j].value[0] == 1):
        temp += (Presences[k,j] * Weights[i,k] * trans * Probs[k,j] * (pt * FN + (1 - pt)))
    temp = 1 - temp
    #########################################################
    
    temp2 = Probs[i, j] * pt * FN + Probs[i, j] * (1-pt)
    
    temp3 = 1-((1-Probs[i, j]) * temp)
    
    temp4 =  temp2 * (1-Presences[i, j]) + Presences[i, j] * temp3
    
    return temp4

def Gekko_obj_func(Probs, Presences):
    n , T = Presences.shape
    obj = 0
    cnt = 0
    for j in range(0, T):
        for i in range(n):
            cnt = cnt + Presences[i, j]
            if j < T - 1:
                obj = obj + Probs[i, j + 1] * Presences[i, j]
            else:
                obj = obj + 2 * Probs[i, j+1] * Presences[i, j]
            
    obj = obj / (n * (T + 1))
    return obj

def Gekko_optimal_solution_without_test(Weights, Vaccine, Group, InitProbs, 
                            time_period, accuracy_of_tests, betau, betav, 
                            min_presense_per_emp_in_week, 
                            min_occupancy_per_day, 
                            max_occupancy_per_day, 
                            test_probability_per_day):
    n, temp = Weights.shape
    #
    #   https://apmonitor.com/wiki/index.php/Main/OptionApmSolver
    #

    '''
    per_model = GEKKO()
    per_model = GEKKO(remote = True, server = 'https://byu.apmonitor.com')
    per_model.solver_options = ['minlp_gap_tol 1.0e-20', \
                    'minlp_maximum_iterations 10000000', \
                    'minlp_max_iter_with_int_sol 500000', 'MAX_TIME 0.0002']
                    
    per_model.solver_options = ['MAX_TIME 0.0002']
    per_model.options.SOLVER = 1
    '''
    
    per_model = GEKKO(remote = False)
    per_model.options.SOLVER = 1
    
    
    # per_model.options.OTOL = 1.0e-8
    # per_model.options.RTOL = 1.0e-8
    # per_model.solver_options = ['objective_convergence_tolerance 1.0e-1',
    #                             'constraint_convergence_tolerance 1.0e-1']
    
    #per_model.options.MAX_TIME = 20
    
    
    min_presense_per_emp_in_week = per_model.Param(value = min_presense_per_emp_in_week)
    min_occupancy_per_day = per_model.Param(value = min_occupancy_per_day)
    max_occupancy_per_day = per_model.Param(value = max_occupancy_per_day)
    test_probability_per_day = per_model.Param(value = test_probability_per_day)
    
    Probs = per_model.Array(per_model.Var, (n, (time_period+1)), integer = False, lb = 0, ub = 1)
    Presences = per_model.Array(per_model.Var, (n , time_period), integer = True, lb = 0, ub = 1)
    
    per_model.Minimize(Gekko_obj_func(Probs, Presences))      # Add this line and remove the folowing two lines
    
    for i in range (n):
        per_model.Equation(Probs[i, 0]  ==  InitProbs[i])

    for i in range (n):
        for j in range (time_period):
            temp = betav * Vaccine[i]+betau * (1-Vaccine[i])
            per_model.Equation(Probs[i, j+1]  ==  Gekko_cons_func(Presences, Probs, temp, Weights, accuracy_of_tests, test_probability_per_day, i, j))

    for i in range(n):
        per_model.Equations([sum(Presences[i,:])  >=  min_presense_per_emp_in_week])
    
    for t in range(time_period):
        per_model.Equations([sum(Presences[:,t])  >=  min_occupancy_per_day])
        per_model.Equations([sum(Presences[:,t])  <=  max_occupancy_per_day])
    
    
    per_model.solve(disp = False)
    #per_model.solver_options = ['max_iter 1000000000', 'max_cpu_time 100000000']
    
    print('Objective val (Gekko): ', per_model.options.OBJFCNVAL)

    Presences, Probs =  convert_Gekko_type_to_data(Presences, Probs)
    return Presences, Probs, per_model.options.OBJFCNVAL

#######################################################################################
#######################################################################################
####  Gurobi
def Gurobi_consfunc(model,trans, Weights,i,j):
    FN=1-model.accuracy_of_tests
    pt=model.test_probability_per_day
    
    # temp = 1
    # for k in range (0,n):
    #     temp=temp * (1-Weights[i,k]*trans*Probs[k,j]*Presences[k,j]*(pt*FN + (1-pt)))
        
    # Linearization
    temp = 0
    for k in range (n):
        if (pe.value(model.Presences[k,j]) == 1):
            temp += (Weights[i,k]*trans*model.Probs[k,j]* 1 *(pt*FN + (1-pt)))
    temp = 1 - temp
    
    temp2=model.Probs[i,j]*pt*FN + model.Probs[i,j]*(1-pt)
    
    temp3=1-((1-model.Probs[i,j]) * temp)
    
    temp4= temp2 * (1-model.Presences[i,j]) + model.Presences[i,j]*temp3
    
    return temp4

def Gurobi_objfunc(Probs,Presences, n, T):
    obj = 0
    cnt = 0
    for j in range(0, T):
        for i in range(n):
            cnt = cnt + Presences[i, j]
            if j<T-1:
                obj = obj + Probs[i, j+1] * Presences[i, j]
            else:
                obj = obj + 2 * Probs[i, j+1] * Presences[i, j]
            
    obj = obj / (n * (T + 1))
    return obj

def Gurobi_solution(Weights, Vaccine, Group, InitProbs,
                            time_period,accuracy_of_tests, betau, betav,
                            min_presense_per_emp_in_week,
                            min_occupancy_per_day,
                            max_occupancy_per_day,
                            test_probability_per_day):
    
    n,_ = Weights.shape
    
    #solver = po.SolverFactory('gurobi') # Other solvers: baron, glpk, pulp, ample, gams
    solver = po.SolverFactory("gurobi", solver_io = 'python')

    model = pe.ConcreteModel()
    
    model.n = pe.Param(initialize = n)
    model.time_period = pe.Param(initialize = time_period)
    
    model.min_occupancy_per_day = pe.Param(initialize = min_occupancy_per_day)
    model.max_occupancy_per_day = pe.Param(initialize = max_occupancy_per_day)
    model.accuracy_of_tests = pe.Param(initialize = accuracy_of_tests)
    model.test_probability_per_day = pe.Param(initialize = test_probability_per_day)
    model.InitProbs = pe.Param(pe.RangeSet(0,n-1),initialize = InitProbs)
    model.Vaccine = pe.Param(pe.RangeSet(0,n-1),initialize = Vaccine)
    #model.Weights=pe.Param(pe.RangeSet(0,n-1),pe.RangeSet(0,n-1), initialize=Weights)
    
    model.Probs = pe.Var(pe.RangeSet(0, n - 1), pe.RangeSet(0, time_period + 1), domain = pe.NonNegativeReals, initialize = 0)
    
    model.Presences = pe.Var(pe.RangeSet(0,n - 1), pe.RangeSet(0, time_period - 1), domain = pe.Binary, initialize = 0)
    
    model.obj = pe.Objective(sense = pe.minimize, expr = Gurobi_objfunc(model.Probs, model.Presences, n, time_period))
    
    model.MyCons = pe.ConstraintList()

    for i in range(n):
        model.MyCons.add(expr = sum(model.Presences[i,:]) >= min_presense_per_emp_in_week)
    
    for t in range(time_period):
        model.MyCons.add(expr = sum(model.Presences[:,t]) >= model.min_occupancy_per_day)
        model.MyCons.add(expr = sum(model.Presences[:,t]) <= model.max_occupancy_per_day)
    
    for i in range (n):
        model.MyCons.add(expr = model.Probs[i,0] == model.InitProbs[i])
        
    for i in range (pe.value(model.n)):
        temp0 = betav * model.Vaccine[i] + betau * (1-model.Vaccine[i])
        for j in range (pe.value(model.time_period)):
            model.MyCons.add(expr = model.Probs[i,j + 1] == 
                Gurobi_consfunc(model, temp0, Weights, i, j))

    model.IntFeasTol = 1
    
    result=solver.solve(model)

    # for i in range(n):
    #     for j in range(time_period):
    #         print(model.Presences[i,j].value, end = ' ')
    #     print('\r\n')
    
    print("\r\nObjective val (Gurobi):", model.obj())
    
    #model.display()
    Presences, Probs = convert_gurobi_type_to_data(model.Presences, model.Probs, n, time_period)
    return Presences, Probs, model.obj()

##############################################################################################
#########  Genetic Algorithm Section (without test)

def GA_sol_initialize(n, T, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day):
    X = np.zeros((n, T))
    
    for t in range(T):
        s = sum(X[:, t])
        while s>max_occupancy_per_day:
            i = randint(0, n-1)
            if X[i, t] == 1:
                X[i, t] = 0
                s = s-1
                
    for t in range(T):
        s = sum(X[:, t])
        while s<min_occupancy_per_day:
            i = randint(0, n-1)
            if X[i, t] == 0:
                X[i, t] = 1
                s = s+1
    
    for i in range(n):
        t = sum(X[i,:])
        while(t<min_presense_per_emp_in_week):
            j = randint(0, T-1)
            if X[i, j] == 0:
                X[i, j] = 1
                t = t+1
    return X

def GA_initialize(popsize, n, T, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day):
    Pop = list()
    for s in range(popsize):
        X = GA_sol_initialize(n, T, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
        Pop.append(copy.copy(X))
    return Pop

def GA_update_best_sol(bestsol, bestobj, Pop, Obj):
    minobj = min(Obj)
    if minobj<bestobj:
        bestobj = minobj
        i = Obj.index(minobj)
        bestsol = copy.copy(Pop[i])
        
    return bestsol, bestobj

def GA_feasibility(X, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day ):
    n, T = X.shape
    penalty=0
    
    for t in range(T):
        s = sum(X[:, t])
        if s > max_occupancy_per_day:
            penalty += (s-max_occupancy_per_day)
        if s < min_occupancy_per_day:
            penalty += (min_occupancy_per_day-s)
     
    for i in range(n):
        s = sum(X[i, :])
        if s < min_presense_per_emp_in_week:
            penalty += (min_presense_per_emp_in_week-s)    

    return penalty

def GA_fitness(X, Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests):
    n, T = X.shape
    FN = 1-accuracy_of_tests
    pt = test_probability_per_day
    
    Probs = np.zeros((n, T+1))
    for i in range(n):
        Probs[i, 0] = InitProbs[i]
    
    for j in range(T):
        for i in range(n):
            trans = betav * Vaccine[i]+betau * (1-Vaccine[i])
            
            temp = 1
            for k in range (n):
                temp = temp *  (1-Weights[i, k] * trans * Probs[k, j] * X[k, j] * (pt * FN + (1-pt)))
            
            # Linearization
            # temp = 0
            # for k in range (n):
            #     temp = temp +  (Weights[i, k] * trans * Probs[k, j] * X[k, j] * (pt * FN + (1-pt)))
            # temp = 1 - temp
            ########################
            
            temp2 = Probs[i, j] * pt * FN + Probs[i, j] * (1-pt)
            temp3 = 1-((1-Probs[i, j]) * temp)
            Probs[i, j+1] =  temp2 * (1-X[i, j]) + X[i, j] * temp3
  
    obj = 0
    cnt = 0
    for j in range(0, T):
        for i in range(n):
            cnt = cnt+X[i, j]
            if j<T-1:
                obj = obj+Probs[i, j+1] * X[i, j]
            else:
                obj = obj+2 * Probs[i, j+1] * X[i, j]
            
    obj = obj/(n * (T+1))
    return obj

def GA_crossover(X, Y):
    n, T = X.shape
    Z1 = np.zeros((n, T))
    Z2 = np.zeros((n, T))
    
    crosspoint = randint(1, n-2)
    
    for i in range(n):
        if i <= crosspoint:
            Z1[i, :] = copy.copy(X[i, :])
            Z2[i, :] = copy.copy(Y[i, :])
        else:
            Z1[i, :] = copy.copy(Y[i, :])
            Z2[i, :] = copy.copy(X[i, :])
    
    return Z1, Z2

def GA_crossover2(X, Y):
    n, T = X.shape
    Z1 = np.zeros((n, T))
    Z2 = np.zeros((n, T))
    
    crosspoint = randint(1, T-2)
    
    for t in range(T):
        if t <= crosspoint:
            Z1[:, t] = copy.copy(X[:, t])
            Z2[:, t] = copy.copy(Y[:, t])
        else:
            Z1[:, t] = copy.copy(Y[:, t])
            Z2[:, t] = copy.copy(X[:, t])
    
    return Z1, Z2

def GA_revise_sol(X, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day):
    n, T = X.shape
    
    for t in range(T):
        s = sum(X[:, t])
        while s>max_occupancy_per_day:
            i = randint(0, n-1)
            if X[i, t] == 1:
                X[i, t] = 0
                s = s-1
                                
    for t in range(T):
        s = sum(X[:, t])
        while s<min_occupancy_per_day:
            i = randint(0, n-1)
            if X[i, t] == 0:
                X[i, t] = 1
                s = s+1
                
    for i in range(n):
        s = sum(X[i, :])
        while s<min_presense_per_emp_in_week:
            t = randint(0, T-1)
            if X[i, t] == 0:
                X[i, t] = 1
                s = s+1            

    return X

def GA_combine(Pop, Obj, Pop2, Obj2):
    Pop3 = list()
    Obj3 = list()
    
    max_bound = max(Obj)
    max_bound = max_bound+max(Obj2)
    
    for s in range(len(Pop)):
        x = min(Obj)
        i = Obj.index(x)
        
        x2 = min(Obj2)
        i2 = Obj2.index(x2)
        
        if(x <= x2):
            Pop3.append(copy.copy(Pop[i]))
            Obj3.append(copy.copy(Obj[i]))
            Obj[i] = max_bound
        else:
            Pop3.append(copy.copy(Pop2[i2]))
            Obj3.append(copy.copy(Obj2[i2]))
            Obj2[i2] = max_bound
        
    return Pop3, Obj3

def GA_mutation(X):
    n, T = X.shape
    X2 = copy.copy(X)
    #for i in range(n):
    i = randint(0, n-1)
    t1 = randint(0, T-1)
    t2 = randint(0, T-1)
    
    temp = X2[i, t2]
    X2[i, t2] = copy.copy(X2[i, t1])
    X2[i, t1] = copy.copy(temp)
    
    i = randint(0, n-1)
    t1 = randint(0, T-1)
    t2 = randint(0, T-1)
    
    temp = X2[i, t2]
    X2[i, t2] = copy.copy(X2[i, t1])
    X2[i, t1] = copy.copy(temp)
    
    return X2

def Genetic_Algorithm(Weights, Vaccine, Group, InitProbs, 
                            time_period, accuracy_of_tests, betau, betav, 
                            min_presense_per_emp_in_week, 
                            min_occupancy_per_day, 
                            max_occupancy_per_day, 
                            test_probability_per_day):
    n, T = Weights.shape
    T = time_period
    popsize = 100
    max_gen = 150
    mutprob = 0.90
    crossprob = 0.80
    Averagelist = list()
    X_axis = list()
    
    bestsol = np.zeros((n, T))
    bestobj = n * T

    Pop = GA_initialize(popsize, n, T, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
    Obj = list()
    
    for s in range (popsize):
        penalty=GA_feasibility(Pop[s], min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
        Obj.append(penalty + GA_fitness(Pop[s], Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests))
    
    bestsol, bestobj = GA_update_best_sol(bestsol, bestobj, Pop, Obj)
    
    for g in range(max_gen):
        Averagelist.append(sum(Obj)/popsize)
        X_axis.append(g)
        
        #print("gen ", g, " ", bestobj, "   ", sum(Obj)/popsize)
        Pop2 = list()
        Obj2 = list()
        
        s = 0
        while s<crossprob * popsize:
            p1 = randint(0, popsize-1)
            temp = randint(0, popsize-1)
            if(Obj[p1]>Obj[temp]):
                p1 = temp
                
            p2 = randint(0, popsize-1)
            temp = randint(0, popsize-1)
            if(Obj[p2]>Obj[temp]):
                p2 = temp
            
            X1, X2 = GA_crossover(Pop[p1], Pop[p2])
            if randint(0, 100)<100 * mutprob:
                X1 = GA_mutation(X1)
            if randint(0, 100)<100 * mutprob:
                X2 = GA_mutation(X2)
                
            X1 = GA_revise_sol(X1, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
            X2 = GA_revise_sol(X2, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
            
            penalty = GA_feasibility(X1, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
            temp1 = penalty + GA_fitness(X1, Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests)
            
            penalty = GA_feasibility(X2, min_presense_per_emp_in_week, min_occupancy_per_day, max_occupancy_per_day)
            temp2 = penalty + GA_fitness(X2, Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests)
            
            Pop2.append(X1)
            Obj2.append(temp1)
            Pop2.append(X2)
            Obj2.append(temp2)
            s = s+2
            
        s = len(Pop2)
        while s<popsize:
            x = min(Obj)
            i = Obj.index(x)
            Pop2.append(copy.copy(Pop[i]))
            Obj2.append(copy.copy(Obj[i]))
            Obj[i] = n * T
            s = s+1
            
        #Pop, Obj = GA_combine(Pop, Obj, Pop2, Obj2)
        Pop = copy.copy(Pop2)
        Obj = copy.copy(Obj2)
        bestsol, bestobj = GA_update_best_sol(bestsol, bestobj, Pop, Obj)
        #i = randint(0, popsize-1)
        #Pop[i] = copy.copy(bestsol)
        #Obj[i] = copy.copy(bestobj)
    
    if (bestobj <= 1):
        print("\r\nObjective Genetic Alg: ", bestobj)
    else:
        print("The Genetic Algorithm cannot find a feasible solution!")
        print("The number of violated constraints are ", int(bestobj))
    #print(bestsol)
    
    # fig, ax = plt.subplots()
    # ax.plot(X_axis, Averagelist, '-b', label = "Ave")
    # plt.show()
    
    Probs = compute_probs(bestsol, Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests)
    return bestsol, Probs, bestobj

def compute_probs(X, Weights, InitProbs, Vaccine, betau, betav, test_probability_per_day, accuracy_of_tests):
    n, T = X.shape
    FN = 1-accuracy_of_tests
    pt = test_probability_per_day
    
    Probs = np.zeros((n, T+1))
    for i in range(n):
        Probs[i, 0] = InitProbs[i]
    
    for j in range(T):
        for i in range(n):
            trans = betav * Vaccine[i]+betau * (1-Vaccine[i])
            temp = 1
            for k in range (n):
                temp = temp *  (1-Weights[i, k] * trans * Probs[k, j] * X[k, j] * (pt * FN + (1-pt)))
            temp2 = Probs[i, j] * pt * FN + Probs[i, j] * (1-pt)
            temp3 = 1-((1-Probs[i, j]) * temp)
            Probs[i, j+1] =  temp2 * (1-X[i, j]) + X[i, j] * temp3
            
    return Probs

#####################################################################################################
#########  Input section 
in_file_path  = 'C:/Users/davood68/Desktop/input_graph.xls'
out_file_directory = 'C:/Users/davood68/Desktop/without_test/scheduling'

Weights, Vaccine, Group = read_input_graph(in_file_path)

Weights, Vaccine, Group = generate_random_graph(n = 70, weightprob = 0.10, vaccinprob = 0.95, groupnum = 3)
 
### weightprob<0.5, With probability "weightprob", a weight will set to 1
### and With probability "weightprob", a weight will set to 0.5
### O.W. (with probability 1-2 * weightprob), a weight will set to 0

draw_physical_graph(Weights, Vaccine)

n, temp = Weights.shape   # n is the number of employees, number of rows minus one

time_period = 5

number_of_weekly_incidences = 300
background_risk = (1 / 7) * (number_of_weekly_incidences / 100000)  # probability of infection outside per day

min_presense_per_emp_in_week = math.ceil(time_period / 2)  # each employee have to present at least "min_presense_per_emp_in_week" days per week

min_occupancy_per_day = 0.40 * n                   # at least "min_number_per_day" of employees have to present at office
max_occupancy_per_day = 0.80 * n                   # at most "max_number_per_day" of employees can present at office per day

test_probability_per_day = 0.2            # probability of an employee do a test voluntary, pr(test)

vaccine_efficiency = 0.85
betau = 0.10                    # transmission probability per contact for unvaccine individuals
betav = (1 - vaccine_efficiency) * betau
    
accuracy_of_tests = 0.75            # FN

# after the Saturday and the Sunday's vacation, employees have a risk of infection od two days
Monday_infection_probability = 1 - (1 - background_risk) * (1 - background_risk)

#Initialization: We assume the employees do a test before starting the week, on monday's morning
# They stay at home, if the result is positive, and come at office otherwise.
# (1-accuracy_of_tests) is the inaccuracy. false negative probability

InitProbs = list()
for i in range (n):
    if Vaccine[i]  ==  1:
        InitProbs.append((1 - vaccine_efficiency) * Monday_infection_probability)
    else:
        InitProbs.append(Monday_infection_probability)


####################################################################
##############  Algorithms Section
elapsed_time_gekko = -1
elapsed_time_gurobi = -1
elapsed_time_genetic = -1

                             # GEKKO
# st = time.time()
# Presences, Probs, obj = Gekko_optimal_solution_without_test(Weights, Vaccine, Group, InitProbs, 
#                             time_period, accuracy_of_tests, betau, betav, 
#                             min_presense_per_emp_in_week, 
#                             min_occupancy_per_day, 
#                             max_occupancy_per_day, 
#                             test_probability_per_day)


# et = time.time()
# elapsed_time_gekko = et - st

# objlinear, objexact = double_check_Obj_Function(Presences, Probs, 
#                          Weights, Vaccine, Group, InitProbs, 
#                            time_period, accuracy_of_tests, betau, betav,test_probability_per_day)

# out_file_path = out_file_directory + '_gekko.xls'
# report_excel_output(Presences, obj, objlinear, objexact, elapsed_time_gekko, out_file_path)

#########################################################################################################

                             # Gurobi
st = time.time()
Presences, Probs, obj = Gurobi_solution(Weights, Vaccine, Group, InitProbs,
                            time_period,accuracy_of_tests, betau, betav,
                            min_presense_per_emp_in_week,
                            min_occupancy_per_day,
                            max_occupancy_per_day,
                            test_probability_per_day)
et = time.time()
elapsed_time_gurobi = et - st

objlinear, objexact= double_check_Obj_Function(Presences, Probs, 
                         Weights, Vaccine, Group, InitProbs, 
                           time_period, accuracy_of_tests, betau, betav,test_probability_per_day)

out_file_path = out_file_directory + '_gurobi.xls'
report_excel_output(Presences, obj, objlinear, objexact, elapsed_time_gurobi, out_file_path)


#######################################################################################################

                             # Genetic

st = time.time()
Presences, Probs, obj = Genetic_Algorithm(Weights, Vaccine, Group, InitProbs, 
                           time_period, accuracy_of_tests, betau, betav, 
                           min_presense_per_emp_in_week, 
                           min_occupancy_per_day, 
                           max_occupancy_per_day, 
                           test_probability_per_day)

et = time.time()
elapsed_time_genetic = et - st

objlinear, objexact = double_check_Obj_Function(Presences, Probs, 
                         Weights, Vaccine, Group, InitProbs, 
                           time_period, accuracy_of_tests, betau, betav,test_probability_per_day)

out_file_path = out_file_directory + '_genetic.xls'
report_excel_output(Presences, obj, objlinear, objexact, elapsed_time_genetic, out_file_path)

#################################################################

# drawing_graph_week(Weights, Presences, Probs, background_risk)

if elapsed_time_gekko >= 0:
    print('\r\n    Gekko Execution  time : ', int(elapsed_time_gekko), 'seconds')
if elapsed_time_gurobi >= 0:
    print('    Gurobi Execution time : ', int(elapsed_time_gurobi), 'seconds')
if elapsed_time_genetic >= 0:
    print('    Genetic Execution time: ', int(elapsed_time_genetic), 'seconds')
