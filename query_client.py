#!/usr/bin/env python3.6

import sys
import rospy
import numpy as np
from rosplan_knowledge_msgs.srv import *
from rosplan_knowledge_msgs.msg import *
from pgmpy.models.BayesianModel import BayesianModel
import pandas as pd
import numpy as np
import warnings
from pgmpy.models import DynamicBayesianNetwork as DBN
from rosplan_dispatch_msgs.msg import EsterelPlanArray
from rosplan_dispatch_msgs.msg import EsterelPlan
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import scipy.misc
from networkx.utils import is_string_like
from networkx.drawing.layout import shell_layout, \
    circular_layout, kamada_kawai_layout, spectral_layout, \
    spring_layout, random_layout, planar_layout
matplotlib.use('TkAgg')



msg = None

class Graph(object):        ## CLASS TO GENERATE The Dynamic bayes network and to find the probability of success of an plan.
    
    def __init__(self):         ####constructor  method initialize the object
        self.plans=[]
        self.final=[]
        self.final1=[]
        self.final2=[]
        self.nodes_len=0
        self.my_dic={}
        self.my_dic1={}
        self.my_dic2={}
        self.received=False
        self.grounded=[]
        self.action_values={}
        self.actions2=0
        self.statenodes=[]
        
        

    def total_plan(self,data): ##### calll back for the the total order plans from oscar.
        msg=data
       
        total_plans=len(msg.esterel_plans)
        self.nodes_len=len(msg.esterel_plans[0].nodes)
        print(total_plans)
        rospy.loginfo("total plan received :" + str(total_plans))
        rospy.loginfo("probability :" + str(len(msg.plan_success_prob)))
        

            
        for i in range(len(msg.esterel_plans)):###for all the total order plans
            new=[]
            for j in range(len(msg.esterel_plans[i].nodes)):#### Actions in each plan 
                new.append(msg.esterel_plans[i].nodes[j].name)
                self.grounded.append(msg.esterel_plans[i].nodes[j].action.name)
                # val.append(msg.esterel_plans[i].nodes[j].action.parameters[i].)
                ### each plan with actions array of array.
                self.my_dic1[i]=(msg.esterel_plans[i].nodes[j].name +" "+ str(msg.esterel_plans[i].nodes[j].action.parameters))
            self.plans.append(new)
            
             
    
        
        


       
        size= len(msg.esterel_plans[0].nodes[0].action.parameters)
        print(size)
        
            
###########  Total order plans actions with parameters ##########################################
        for i in range(len(msg.esterel_plans)):
            for j in range(len(msg.esterel_plans[i].nodes)):
                self.my_dic[j]= msg.esterel_plans[i].nodes[j].name

                for k in range(len(msg.esterel_plans[i].nodes[j].action.parameters)): 
                    self.my_dic[j] += " " + msg.esterel_plans[i].nodes[j].action.parameters[k].value
                    # print(self.my_dic[j])


        

        for i in range(len(self.my_dic)):
            # if i % 2==0:
            self.final.append(self.my_dic[i])
            self.final1.append(self.final[i].split())
            self.final2.append('#'.join(self.final1[i])+'%'+str(i))
            

            

        # list1=plan+val1
        self.received=True
        # self.final.append(str(' '.join(plan)))
        # print("totalplans",self.action_values)
        print("totalplans",self.final2)
        # print("grounded actions",self.grounded)
        # print("totalplans",self.my_dic)
        # print("dictionary with parameters of total plans",self.my_dic)    
        # print("PLAN",plan)    
        # print("values",val1)
        

    def dump(self):
        f= open("predicates.txt","w+")
        print("avinash")
        f.write("state nodes for each layer \n")
        for i in range(len(self.statenodes)):
            f.write("Layer No :"+ str(i)+"\n")
            for j in range(len(self.statenodes[i])):
                    f.write( self.statenodes[i][j]+"\n")
                   
        f.close()

    def parents(self):
        f= open("parents_and_children.txt","w+")
        f.write("Parents nodes for each layer \n")
        for i in range(len(self.statenodes)):
            for j in range(len(self.statenodes[i])):
                f.write("Node_name :" +self.statenodes[i][j] +"\n")
                parent=self.model.get_parents(self.statenodes[i][j])
                f.write("Parent:"+ str(parent)+ "\n")
                f.write("\n")
            
        for i in range(self.actions2):
            f.write("Parents for actions \n")
            f.write("Node_name :" +self.final2[i] +"\n")
            parent=self.model.get_parents(self.final2[i])
            f.write("Parent:"+ str(parent)+ "\n")
            f.write("\n")

        for i in range(self.actions2):
            f.write("Children for actions \n")
            f.write("Node_name :" +self.final2[i]+"\n")
            child=self.model.get_children(self.final2[i])
            f.write("Children:"+ str(child)+ "\n")
            f.write("\n")


    
        

    def call_service(self):
        
#### subscribing to the node to get the total order plans.####################################
        rospy.Subscriber("/csp_exec_generator/valid_plans", EsterelPlanArray, self.total_plan)

        print ("Waiting for service")

        rospy.wait_for_service('/rosplan_knowledge_base/domain/name')
        rospy.wait_for_service('/rosplan_knowledge_base/domain/types')
        # rospy.wait_for_service('/rosplan_knowledge_base/domain/functions')
        rospy.wait_for_service('/rosplan_knowledge_base/domain/operators')
        rospy.wait_for_service('/rosplan_knowledge_base/domain/operator_details')
        # rospy.wait_for_service('/rosplan_knowledge_base/domain/predicate_details')
        rospy.wait_for_service('/rosplan_knowledge_base/domain/predicates')
        rospy.wait_for_service('/rosplan_knowledge_base/state/propositions')
        # rospy.wait_for_service('/rosplan_knowledge_base/state/timed_knowledge')
        # rospy.wait_for_service('/rosplan_knowledge_base/query_state')
        # # try:
        print ("Calling Service")

        domain_name = rospy.ServiceProxy('/rosplan_knowledge_base/domain/name', GetDomainNameService)
        domain_types = rospy.ServiceProxy('/rosplan_knowledge_base/domain/types',GetDomainTypeService)
        # domain_functions = rospy.ServiceProxy('/rosplan_knowledge_base/domain/functions', GetDomainAttributeService)
        domain_operatordetails = rospy.ServiceProxy('/rosplan_knowledge_base/domain/operator_details', GetDomainOperatorDetailsService)
        domain_operators = rospy.ServiceProxy('/rosplan_knowledge_base/domain/operators', GetDomainOperatorService)
        domain_predicates = rospy.ServiceProxy('/rosplan_knowledge_base/domain/predicates', GetDomainAttributeService)
        problem_initialstate = rospy.ServiceProxy('/rosplan_knowledge_base/state/propositions',GetAttributeService)
        # domain_timedknowledge = rospy.ServiceProxy('/rosplan_knowledge_base/state/timed_knowledge',GetAttributeService)
        # query_proxy = rospy.ServiceProxy('/rosplan_knowledge_base/query_state', KnowledgeQueryService)


        resp1 = domain_name()
        resp2 = domain_types()
        
        resp5 = domain_operators()
        # resp3 = domain_functions()
        
        # resp4.op.formula.typed_parameters[0].key
        # resp4 = domain_operatordetails(act)
        resp7 = domain_predicates()
        resp8 = problem_initialstate()
        
        # print("resp7", resp7)

        
        # print ("action_names",resp4.op.formula.name)


        #actions
        act_nodes=[]
        actions=len(resp5.operators)
        print("actions",actions)
        for i in range(actions):
            act_nodes.append(str(resp5.operators[i].name)) 
            
        print("action_nodes",act_nodes)


        print("waiting for the message")
        while self.received is False :
            continue

        TotalPlans=self.plans
        # finalplans=self.final
        finalplans1=self.final2
        
        

        
        
        # # print(TotalPlans[0][:])
        ncol=len(TotalPlans[0])
        # print("coloums",ncol)

        

        
        durative=[]
        Actions_wth_param=[]
        for i in range(1):
            for j in range(ncol):
                Actions_wth_param.append(TotalPlans[i][j])
                if j % 2 == 0:
                    durative.append(TotalPlans[i][j])
        actions1=len(durative)#### Not every action 17
        self.actions2=len(Actions_wth_param)### every action 34
        print("durative actions",actions1)
        print("durative actions",durative)
        # print("actions_plan",Actions_wth_param)
        grounded=[]
        grounded1=[]
        for i in range(len(self.grounded)):
            grounded1.append(self.grounded[i])
            
            if i%2!=0:
                grounded.append(self.grounded[i])
                
        # print("grounded",grounded)        
        # print("grounded",grounded1)  

        ######## start actions and end actions from a plan

        start_actions=[]
        end_actions=[]
        for i in range(len(self.final)):
            if i%2==0:
                start_actions.append(self.final[i])
            else:
                end_actions.append(self.final[i])

        start_actions1=[]
        end_actions1=[]
        for i in range(len(self.final2)):
            if i%2==0:
                start_actions1.append(self.final2[i])
            else:
                end_actions1.append(self.final2[i])
        
                        
        
        



        # print("start_actions",start_actions)
        # print("end_actions",end_actions )
        start_cond=[]
        end_cond=[]
        overall_cond=[]
        at_start_eff=[]
        at_start_deleff=[]
        at_end_addeff=[]
        at_end_deleff=[]
        operator_details=[]
        operator_detail=[]
        values=[]
        names=[]
        start=[]
        end=[]
        overall=[]
        start_param=[]
        end_param=[]
        overall_param=[]
        start_eff=[]
        start_del=[]
        end_eff=[]
        end_del=[]
        add_param=[]
        del_param=[]
       
        end_del_param=[]
        variable=[]
        grounded_parameteres=[]
        start1=[]
        res=[]
        # start_strings =[]
        end_strings =[]
        overall_strings =[]
        res2 =[]
        str1=[]
        str2=[]
        str3=[]
        str4=[]
        str5=[]
        str6=[]
        str7=[]
        all_set=set()
        strt_c={}

        self.model=BayesianModel()	
        self.dbn=DBN()

    ################### start actions ###############################################


        for i in range(self.actions2): ########  action nodes from total order plan ##############################
            self.model.add_node(finalplans1[i]+'%'+str(i))



    ################### ACTION PREDICATES FIELD########################################################


        for i in range(actions1):
            start_strings =[]
            end_strings=[]
            overall_strings=[]
            start_param=[]
            end_param=[]
            add_param=[]
            end_del_param=[]
            resp4 =domain_operatordetails(grounded[i])
            operator_details.append(str(resp4.op.formula.name))
            # model.add_node(finalplans[i])
#################### Parameters of the action ##########################################################
            size=len(resp4.op.formula.typed_parameters)
            # print(size)
            for j in range(size):
                operator_details[i] =operator_details[i]+ " "+ resp4.op.formula.typed_parameters[j].key

            detail = operator_details[i].split()
            action_name = detail[0]

            var_Dic = dict(zip(detail,start_actions[i].split()))
            del var_Dic[action_name]
            # var_Dic=str(var_Dic)
            # print("dictionary",str(var_Dic))
            variable.append(var_Dic)    #####Dictionary getting the variables for replacement in the preconditions and effects 

    ########## at_start conditions###################################################################
            size=len(resp4.op.at_start_simple_condition)
            # print("preconditions length",size)
            for z in range(size):
                start=[]
                if size!=0 :
                    start.append(str(resp4.op.at_start_simple_condition[z].name))
                    start1.append(str(resp4.op.at_start_simple_condition[z].name))
                for j in range(len(resp4.op.at_start_simple_condition[z].typed_parameters)):
                    start.append(str(resp4.op.at_start_simple_condition[z].typed_parameters[j].key))
                    start1.append(str(resp4.op.at_start_simple_condition[z].typed_parameters[j].key))

                    update=[var_Dic.get(key) for key in start]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None]
                    # print(res_index)
                    # res = start
                    for idx,val in enumerate(res_index):
                        start[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).
                       
                        
                str1.append('#'.join(start)) #+'%'+str(0))
               
                start_strings.append('#'.join(start))#+'%'+str(0))
            start_cond.append(start_strings)
               
            


               
    ########## at_end conditions#####################################################################
            size=len(resp4.op.at_end_simple_condition)
            # print("nnnnn",size)
            for z in range(size):
                end=[]
                if size!=0:
                    end.append(str(resp4.op.at_end_simple_condition[z].name))
                for j in range(len(resp4.op.at_end_simple_condition[z].typed_parameters)):
                    end.append(str(resp4.op.at_end_simple_condition[z].typed_parameters[j].key)) 

                    update=[var_Dic.get(key) for key in end]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)
                    # res1 = end

                    for idx,val in enumerate(res_index):
                        end[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).

                str2.append('#'.join(end))# +'%'+str(0))
                end_strings.append('#'.join(end))# +'%'+str(i))
            end_cond.append(end_strings)
            
                          

    ########## overall conditions####################################################################
            size=len(resp4.op.over_all_simple_condition)
            # print("nnnnn",size)
            for z in range(size):
                overall=[]
                if size!=0:
                    overall.append(str(resp4.op.over_all_simple_condition[z].name))
                for j in range(len(resp4.op.over_all_simple_condition[z].typed_parameters)):
                    overall.append(str(resp4.op.over_all_simple_condition[z].typed_parameters[j].key))

                    update=[var_Dic.get(key) for key in overall]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)

                    for idx,val in enumerate(res_index):
                        overall[val] = var_update[idx] #### replacement done variables with parameters

                str3.append('#'.join(overall))#+'%'+str(0))
                overall_strings.append('#'.join(overall))#+'%'+str(i))
            overall_cond.append(overall_strings)
   

    ########## at_start_add_effects  ###################################################################
            size=len(resp4.op.at_start_add_effects)
            # print("nnnnn",size)
            for z in range(size):
                start_eff=[]
                if size!=0:
                    start_eff.append(str(resp4.op.at_start_add_effects[z].name))
                for j in range(len(resp4.op.at_start_add_effects[z].typed_parameters)):
                    start_eff.append(str(resp4.op.at_start_add_effects[z].typed_parameters[j].key))

                    update=[var_Dic.get(key) for key in start_eff]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)
                    # res1 = end

                    for idx,val in enumerate(res_index):
                        start_eff[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).

                str4.append('#'.join(start_eff))# +'%'+str(0))
                start_param.append('#'.join(start_eff))
            at_start_eff.append(start_param)

    ########## at_start_DEL_effects  ################################################################
            size=len(resp4.op.at_start_del_effects)
            # print("at start del effects",size)
            for z in range(size):
                start_del=[]
                if size!=0:
                    start_del.append(str(resp4.op.at_start_del_effects[z].name))
                for j in range(len(resp4.op.at_start_del_effects[z].typed_parameters)):
                    start_del.append(str(resp4.op.at_start_del_effects[z].typed_parameters[j].key))

                    update=[var_Dic.get(key) for key in start_del]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)
                    # res1 = end

                    for idx,val in enumerate(res_index):
                        start_del[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).

                str5.append('#'.join(start_del))# +'%'+str(0))
                end_param.append('_'.join(start_del))#+'%'+str(i))
            at_start_deleff.append(end_param)
  

                
    ########## at_end_add_effects  #####################################################################
            size=len(resp4.op.at_end_add_effects)
            # print("nnnnn",size)
            for z in range(size):
                end_eff=[]
                if size!=0:
                    end_eff.append(str(resp4.op.at_end_add_effects[z].name))
                for j in range(len(resp4.op.at_end_add_effects[z].typed_parameters)):
                    end_eff.append(str(resp4.op.at_end_add_effects[z].typed_parameters[j].key))

                    update=[var_Dic.get(key) for key in end_eff]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)
                    # res1 = end

                    for idx,val in enumerate(res_index):
                        end_eff[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).

                str6.append('#'.join(end_eff)) #+'%'+str(0))
                add_param.append('#'.join(end_eff))#+'%'+str(i))
            at_end_addeff.append(add_param)


    ########## at_end_del_effects  #######################################################################
            size=len(resp4.op.at_end_del_effects)
            # print("nnnnn",size)
            for z in range(size):
                end_del=[]
                if size!=0:
                    end_del.append(str(resp4.op.at_end_del_effects[z].name))
                for j in range(len(resp4.op.at_end_del_effects[z].typed_parameters)):
                    end_del.append(str(resp4.op.at_end_del_effects[z].typed_parameters[j].key))    

                    update=[var_Dic.get(key) for key in end_del]
                    var_update = [i for i in update if i is not None]
                    # print("l_update",var_update)
                    res_index = [i for i in range(len(update)) if update[i] != None] # other than Nones positions
                    # print(res_index)
                    # res1 = end

                    for idx,val in enumerate(res_index):
                        end_del[val] = var_update[idx] #### replacement done variables with parameters dictionary(var_dic).

                str7.append('#'.join(end_del))# +'%'+str(0))
                end_del_param.append('#'.join(end_del))# +'%'+str(i))
            at_end_deleff.append(end_del_param)

      
        
        # result=dict(zip(operator_details[0].split(),self.final[0].split()))
        
        # result=set(result)
        # print("result",variable)####### Dictionary of parameters for replacement.
        print("final",res)
        # print("start_conditions",start_strings)
        joinedlist=str1+str2+str3+str4+str5+str6+str7      
        # joinedlist=start_strings+end_strings+overall_strings+start_param+end_param+add_param+end_del_param
        joinedlist1=start_cond+end_cond+overall_cond+start_eff+end_eff+at_start_eff+at_start_deleff+at_end_addeff+at_end_deleff
        # print("all set union",joinedlist1)
        # print("all set ",len(joinedlist))

        
        
        # print("new",new)
        # print("operator_details",resp4)
        # print("start_cond",start_cond)
        # print(values)
        # print("parameters",operator_details)
        
        
        print("end_conditions names",end)
        print("overall_conditions names",overall)
        # print("start_conditions parameters",start_param)
        # # print("end_conditions parameters",end_param)
        # # print("overall_conditions parameters",overall_param)
        # print("at_start_add_effects_names",start_eff)
        # # print("at_start_add_effects_parameters",add_param)
        # print("at_start_delete_effects_names",start_del)
        # # print("at_start_delete_effects_parameters",del_param)
        # print("at_end_add_effects_names",end_eff)
        # # print("at_end_add_effects",end_param)
        # print("at_end_del_effects_names",end_del)
        # print("at_end_del_effects",end_del_param)

      

    
        
        

       

        
        length= (len(resp8. attributes))
        # print(length)
       
        initialNodes=[]
        # self.statenodes=[]
    
        
        strings=[]
        
############## Initial nodes #######################################################################################
        
        
        for i in range(length): 
            initialNodes.append(str(resp8. attributes[i].attribute_name))
            # key.append(resp8. attributes[i].values[0].value)
            strings.append(resp8. attributes[i].attribute_name)
            value_size = len(resp8. attributes[i].values)
            for j in range(value_size):
                strings[i] = strings[i] + "# " + resp8.attributes[i].values[j].value ### all the initial predicates from problems
            # initial.append(initialNodes[i]+'#'+key[i])

        # print("initialnodes",str(initialNodes))     
        # print("Initial_nodes",strings)#initial nodes from problem pddl
        
        




######### state nodes in each transition of actions#################################################################

        for i in range(self.actions2+1):
            states=[]
            for nodes in joinedlist:
                
                states.append(str(nodes)+'%'+str(i))

            self.statenodes.append(states)
        Graph.dump(self)	#i+=1
            
        # print("state_nodes",self.statenodes)# state_nodes dbn
#######################################################################################################################
        for i in range(len(self.statenodes)):
            for j in range(len(self.statenodes[i])):
                self.model.add_node(self.statenodes[i][j])############  All predicates  nodes from Domain file ########################
            

    ######state transitions with the initial nodes###########################################
        
        for i in range(self.actions2):
            for j in range(len(self.statenodes[i])):
                self.model.add_edge(self.statenodes[i][j], self.statenodes[i+1][j])
                
                

    ########################################################################################


    ### drawing edges from predicates to actions....########################################

        for i in range(actions1):
            for j in range(len(start_cond[i])):
               self.model.add_edge(start_cond[i][j]+'%' +str(i),start_actions1[i])
        # #         model.add_edge(start_cond[i][j],self.final[i])



        for i in range(actions1):
            for j in range(len(end_cond[i])):
               self.model.add_edge(end_cond[i][j]+'%' +str(i),end_actions1[i])



        for i in range(actions1):
            for j in range(len(overall_cond[i])):
                self.model.add_edge(overall_cond[i][j]+'%' +str(i+1),end_actions1[i])
        


        for i in range(actions1):
            for j in range(len(at_start_eff[i])):
                self.model.add_edge(start_actions1[i],at_start_eff[i][j]+'%' +str(i+1))

        for i in range(actions1):
            for j in range(len(at_start_deleff[i])):
                self.model.add_edge(start_actions1[i],at_start_deleff[i][j]+'%' +str(i+1))


        for i in range(actions1):
            for j in range(len(at_end_addeff[i])):
                self.model.add_edge(end_actions1[i],at_end_addeff[i][j]+'%' +str(i+2))
        

        for i in range(actions1):
            for j in range(len(at_end_deleff[i])):
                self.model.add_edge(end_actions1[i],at_end_deleff[i][j]+'%' +str(i+2))



   

####################################################################################################################
    ###### DBN Graph ########################

        Graph.parents(self)
        
        np.warnings.filterwarnings('ignore')
        warnings.simplefilter(action="ignore",category=RuntimeWarning)
        # nx.draw(self.model, with_labels=True)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)       
        plt.axis('off')
        x=self.model.nodes()
        
        print(len(x))
        
        # plt.show()

    


        rospy.spin()


if __name__ == "__main__":

    rospy.init_node('query_client_node', anonymous=True)
    Graph().call_service()
    # sys.exit(1)
