# -- coding: utf-8 --
import ray

class AgentProxy(object):

    def __init__(self, agent,copy_agent=None):
        self.agent = agent
        self.copy_agent = copy_agent
       
    def choseAction(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.choseAction.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.choseAction.remote(*args , **kwargs))

    def train(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.train.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.train.remote(*args , **kwargs))
    
    def learn_from_experience(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.learn_from_experience.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.learn_from_experience.remote(*args , **kwargs))

    def memory(self, *args,**kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.memory.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.memory.remote(*args , **kwargs))
   
    def store_experience(self,*args,**kwargs):
        ray.get(self.agent.store_experience.remote(*args, **kwargs))

    def update(self,*args,**kwargs):
        ray.get(self.agent.update.remote(*args, **kwargs))
    
    def update_target_model(self,*args,**kwargs):
        ray.get(self.agent.update_target_model.remote(*args, **kwargs))

    def _updateModel(self,*args, **kwargs):
        ray.get(self.agent._updateModel.remote(*args, **kwargs))

    def render(self,*args, **kwargs):
        ray.get(self.agent.render.remote(*args, **kwargs))

    def save(self,*args,**kwargs):
        ray.get(self.agent.save.remote(*args, **kwargs))

    def load(self,*args,**kwargs):
        ray.get(self.agent.load.remote(*args, **kwargs))

    def select_action(self, *args,**kwargs):
        ray.get(self.agent.select_action.remote(*args , **kwargs))
    
    def take_action(self, *args, **kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.take_action.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.take_action.remote(*args , **kwargs))

    def learn(self, *args, **kwargs):
        if self.copy_agent != None:
            return ray.get(self.copy_agent.learn.remote(*args , **kwargs))
        else:
            return ray.get(self.agent.learn.remote(*args , **kwargs))