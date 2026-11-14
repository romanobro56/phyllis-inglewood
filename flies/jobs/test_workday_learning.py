"""Small integration checks of the real core neural path, not performance claims."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from core.brain import FlyBrain,save_brain
from core.train import rollout,settings
from flies.jobs.task import JobsFly
from flies.jobs.learning import reinforce,save_training,Curriculum
from flies.jobs.policy import digest,save_model,load_model,checkpoint_path
from flies.jobs.sim_state import SimulationState


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        self.brain=FlyBrain('tiny',64,8,'jobs')

    def test_reward_updates_neural_parameters_but_not_wiring_signs_or_readout(self):
        class Small(JobsFly):
            def __init__(self):super().__init__(level=1);self.max_steps=16
        before=digest(self.brain)
        fixed={k:v.clone() for k,v in self.brain.named_buffers()}
        _,transitions=rollout(Small,8,np.random.default_rng(3),'brain',self.brain,'cpu')
        cfg=settings(JobsFly,{})
        opt=torch.optim.Adam(self.brain.parameters(),lr=cfg['lr'])
        reinforce(self.brain,opt,transitions,cfg)
        self.assertNotEqual(before,digest(self.brain))
        for k,v in self.brain.named_buffers():self.assertTrue(torch.equal(fixed[k],v),k)

    def test_checkpoint_generation_restores_weights_optimizer_rng_and_curriculum(self):
        cfg=settings(JobsFly,{})
        opt=torch.optim.Adam(self.brain.parameters(),lr=cfg['lr'])
        class Small(JobsFly):
            def __init__(self):super().__init__(level=1);self.max_steps=8
        _,transitions=rollout(Small,4,np.random.default_rng(2),'brain',self.brain,'cpu')
        reinforce(self.brain,opt,transitions,cfg)
        rng=np.random.default_rng(14);world=SimulationState();world.drafts['marker']='persistent simulation state'
        curriculum=Curriculum(level=3,streak=1)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            save_training(out,self.brain,opt,world,rng,curriculum,12,192,[],cfg)
            path=checkpoint_path(out)
            restored=load_model(out,JobsFly)
            self.assertEqual(digest(restored),digest(self.brain))
            state=torch.load(path.parent/'training.pt',weights_only=False)
            self.assertEqual(state['curriculum'],curriculum.snapshot())
            self.assertEqual(state['episodes'],192);self.assertEqual(state['updates'],12)
            again=np.random.default_rng();again.bit_generator.state=state['numpy_rng']
            self.assertEqual(rng.integers(100000),again.integers(100000))
            self.assertEqual(SimulationState.load(path.parent/'simulation_state.json').drafts['marker'],'persistent simulation state')
            restored_opt=torch.optim.Adam(restored.parameters(),lr=cfg['lr'])
            restored_opt.load_state_dict(state['optimizer'])
            self.assertTrue(state['optimizer']['state'])
            for key,values in opt.state_dict()['state'].items():
                for name,value in values.items():
                    self.assertTrue(torch.equal(value,restored_opt.state_dict()['state'][key][name]))
            reinforce(self.brain,opt,transitions,cfg)
            reinforce(restored,restored_opt,transitions,cfg)
            self.assertEqual(digest(self.brain),digest(restored))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA GPU unavailable on this local host')
    def test_cuda_forward_backward(self):
        self.brain=self.brain.to('cuda')
        obs=torch.full((4,64),.5,device='cuda')
        before=digest(self.brain)
        opt=torch.optim.Adam(self.brain.parameters(),lr=.0005)
        logits,_=self.brain(obs,torch.ones((4,8),dtype=torch.bool,device='cuda'))
        (-logits.log_softmax(-1)[:,0].mean()).backward();opt.step()
        self.assertNotEqual(before,digest(self.brain))

    def test_legacy_or_missing_schema_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt';save_brain(self.brain,path,JobsFly)
            with self.assertRaises(ValueError):load_model(path,JobsFly)
            save_model(self.brain,path,JobsFly)
            self.assertEqual(digest(self.brain),digest(load_model(path,JobsFly)))


if __name__=='__main__':unittest.main()
