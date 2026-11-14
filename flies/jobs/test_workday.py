"""Oracle-assisted fixtures test mechanics ONLY; never used as learning evidence."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from flies.jobs.contract import encode, SCHEMA
from flies.jobs.workday import SimulatedWorkdayEnvironment
from flies.jobs.sim_state import SimulationState, SecretVault
from flies.jobs.workday_forms import generate, fingerprint
from flies.jobs.learning import Curriculum


class FixtureDriver:
    """Test-only deterministic fixture arranger. Production never imports this class."""
    def __init__(self, env, mirror=None):
        self.env,self.mirror=env,mirror
        self.rng=np.random.default_rng(999)

    def act(self,predicate):
        for _ in range(100):
            choices=self.env.available_actions()
            index=next((i for i,a in enumerate(choices) if predicate(a)),None)
            if index is not None:
                if self.mirror:self.mirror.execute(self.env,index)
                result=self.env.step(index,self.rng)
                if self.mirror:
                    self.mirror.sync(self.env,result[3],policy='TEST_FIXTURE_NOT_LEARNING',checkpoint='none')
                    self.mirror.check(self.env)
                return result
            more=next((i for i,a in enumerate(choices) if a.kind=='more'),None)
            if more is None:break
            if self.mirror:self.mirror.execute(self.env,more)
            info=self.env.step(more,self.rng)[3]
            if self.mirror:self.mirror.sync(self.env,info);self.mirror.check(self.env)
        raise AssertionError('fixture action unavailable')

    def kind(self,kind,**kwargs):
        return self.act(lambda a:a.kind==kind and all(getattr(a,k)==v for k,v in kwargs.items()))

    def button(self,button):
        if self.env.mode=='edit':self.kind('cancel')
        return self.kind('click',target='button_'+button)

    def enter(self,field,value,ref=None):
        self.kind('focus',target=field['id'])
        if field['kind'] in ('select','combobox','radio'):
            self.kind('choose',value=value)
        else:
            if self.env.level>3:
                if ref and ref.startswith('password_'):
                    group='credentials'
                else:
                    group=next(g for g,record in self.env.catalogs.items() if g!='credentials' and value in record.values())
                self.kind('catalog',ref=group)
            if ref:
                self.kind('enter',ref=ref)
            else:self.kind('enter',value=value)
        self.kind('cancel')

    def auth(self,register=False,credential='password_A'):
        wanted='register' if register else 'login'
        if self.env.auth_view!=wanted:self.button(wanted)
        self.enter(self.env.auth_fields[0],self.env.email)
        self.enter(self.env.auth_fields[1],'',ref=credential)
        if register:self.enter(self.env.auth_fields[2],'',ref=credential)
        return self.button(wanted)

    def complete(self):
        for _ in range(30):
            env=self.env
            if env.done:return
            if env.stage=='auth':
                self.auth(register=not bool(env.state.account(env.tenant,env.email)))
            elif env.stage=='resume':
                self.kind('focus',target='resume_upload');self.kind('upload',ref='resume_pdf');self.kind('cancel');self.button('next')
            elif env.stage=='review':
                self.button('review');self.button('submit');return
            else:
                if env.stage in ('experience','education'):
                    while env.added[env.stage]<len(env.scenario['sections'][env.stage]):self.button('add_entry')
                for field in list(env.current_fields()):
                    expected=env._validator.expected[field['id']]
                    if expected is not None and field['value']!=expected:self.enter(field,expected)
                self.button('next')
        raise AssertionError('fixture did not terminate')


def known_full_fixture(level=9, state=None, vault=None):
    env=SimulatedWorkdayEnvironment(level=level,state=state,vault=vault,trace=True)
    for seed in range(500):
        env.reset(np.random.default_rng(seed))
        all_fields=[]
        for name,group in env.scenario['sections'].items():
            all_fields.extend([f for entry in group for f in entry] if name in ('experience','education') else group)
        if not env.scenario['verification_required'] and not env.scenario['seed_duplicate'] and not any(f['required'] and env._validator.expected[f['id']] is None for f in all_fields):
            return env,seed
    raise AssertionError('no usable full fixture')


class WorkdayTests(unittest.TestCase):
    def test_oracle_cannot_change_public_view_or_actions(self):
        env,_=known_full_fixture()
        before=env.public_view();mask=env.mask();obs=env.obs()
        env._validator.expected={k:'unreachable answer' for k in env._validator.expected}
        self.assertEqual(before,env.public_view());self.assertEqual(mask,env.mask())
        np.testing.assert_array_equal(obs,env.obs())
        self.assertNotIn('expected',json.dumps(before))

    def test_contract_is_64_finite_floats_and_8_structural_slots(self):
        for level in range(1,11):
            env=SimulatedWorkdayEnvironment(level=level)
            for seed in range(5):
                env.reset(np.random.default_rng(seed))
                self.assertEqual(env.obs().shape,(64,));self.assertTrue(np.isfinite(env.obs()).all())
                self.assertEqual(len(env.mask()),8)
                for i,a in enumerate(env.public_view()['actions']):self.assertEqual(env.mask()[i],a is not None)

    def test_same_action_oracle_difference_affects_reward_not_next_sensors(self):
        env=SimulatedWorkdayEnvironment(level=1,trace=True);env.reset(np.random.default_rng(10))
        driver=FixtureDriver(env);field=env.current_fields()[0];expected=env._validator.expected[field['id']]
        driver.kind('focus',target=field['id'])
        other=copy.deepcopy(env);other._validator.expected[field['id']]='impossible'
        i=next(i for i,a in enumerate(env.available_actions()) if a.value==expected)
        first=env.step(i,driver.rng);second=other.step(i,driver.rng)
        self.assertNotEqual(first[1],second[1]);np.testing.assert_array_equal(first[0],second[0])

    def test_unnecessary_review_is_rejected_without_ending_exploration(self):
        env=SimulatedWorkdayEnvironment(level=1);env.reset(np.random.default_rng(1))
        _,reward,done,info=FixtureDriver(env).kind('review')
        self.assertFalse(done);self.assertLess(reward,0);self.assertEqual(env.completed,0)

    def test_account_creation_login_credentials_and_tenant_isolation(self):
        env,_=known_full_fixture(level=4)
        env.scenario['seed_existing']=False
        env.state.accounts.clear()
        env._move(env.scenario['stages'].index('auth'))
        driver=FixtureDriver(env)
        driver.auth(register=True,credential='password_B')
        self.assertEqual(env.account_created,1)
        self.assertFalse(env.state.login('another-tenant',env.email,env.vault.resolve(env.tenant,'password_B')))
        self.assertTrue(env.state.login(env.tenant,env.email,env.vault.resolve(env.tenant,'password_B')))
        env.authenticated=False;env._move(env.scenario['stages'].index('auth'))
        info=driver.auth(register=False,credential='password_A')[3]
        self.assertEqual(info['event'],'validation_error')
        info=driver.auth(register=False,credential='password_B')[3]
        self.assertEqual(info['event'],'login_success')
        def leaves(value):
            if isinstance(value,dict):return [s for v in value.values() for s in leaves(v)]
            if isinstance(value,list):return [s for v in value for s in leaves(v)]
            return [value] if isinstance(value,str) else []
        strings=leaves([env.snapshot(),env.state.snapshot(),env.traces])
        for ref in ('password_A','password_B','password_weak'):
            secret=env.vault.resolve(env.tenant,ref)
            self.assertFalse(any(secret==s or (len(secret)>=8 and secret in s) for s in strings),'Credential material leaked into public or saved data')

    def test_existing_account_and_password_errors(self):
        env,_=known_full_fixture(level=4);env.state.accounts.clear()
        env._move(env.scenario['stages'].index('auth'));driver=FixtureDriver(env)
        self.assertEqual(driver.auth(register=True,credential='password_C')[3]['event'],'validation_error')
        self.assertIn('Password needs',' '.join(env.messages))
        driver.auth(register=True)
        env.authenticated=False;env._move(env.scenario['stages'].index('auth'))
        self.assertEqual(driver.auth(register=True)[3]['event'],'validation_error')
        self.assertIn('already exists',' '.join(env.messages))

    def test_full_workflow_fixture_validation_and_duplicate_receipt(self):
        env,_=known_full_fixture()
        env.max_steps=3000
        driver=FixtureDriver(env);driver.complete()
        self.assertEqual(env.completed,1);self.assertTrue(env.done)
        self.assertEqual(env._validation(whole=True),[])
        self.assertEqual(len(env.state.applications),1)
        self.assertEqual(env.uploaded,1)
        self.assertGreater(env.added['experience'],0);self.assertGreater(env.added['education'],0)
        # A second submit can never earn a second completion reward.
        env.done=False
        _,reward,_,info=driver.button('submit')
        self.assertEqual(info['event'],'duplicate_submission');self.assertLess(reward,0)
        self.assertEqual(len(env.state.applications),1)

    def test_removing_an_entry_discards_values_without_refunding_rewards(self):
        env,_=known_full_fixture(level=6);env._move(env.scenario['stages'].index('experience'))
        driver=FixtureDriver(env);driver.button('add_entry')
        field=next(f for f in env.current_fields() if env._validator.expected[f['id']] is not None and f['kind'] not in ('date',))
        driver.enter(field,env._validator.expected[field['id']])
        self.assertTrue(field['value'])
        self.assertIn('field:'+field['id'],env.paid)
        env.reviewed=True;driver.button('remove_entry')
        self.assertEqual(field['value'],'');self.assertFalse(env.reviewed)
        self.assertIn('field:'+field['id'],env.paid)
        driver.button('add_entry');self.assertEqual(field['value'],'')

    def test_incomplete_submission_and_incorrect_optional_information_rejected(self):
        env,_=known_full_fixture();env._move(env.scenario['stages'].index('review'));env.reviewed=True
        _,reward,_,info=FixtureDriver(env).button('submit')
        self.assertEqual(info['event'],'submission_rejected');self.assertLessEqual(reward,-10)
        self.assertEqual(env.completed,0)

    def test_upload_wrong_file_and_parser_recovery(self):
        env,_=known_full_fixture(level=5);driver=FixtureDriver(env)
        env._move(env.scenario['stages'].index('auth'))
        driver.auth(register=not bool(env.state.account(env.tenant,env.email)))
        env._move(env.scenario['stages'].index('resume'))
        env.scenario['parse']='corrupt'
        driver.kind('focus',target='resume_upload')
        self.assertLess(driver.kind('upload',ref='other_document')[1],0)
        self.assertEqual(env.uploaded,0)
        driver.kind('upload',ref='resume_pdf');self.assertEqual(env.uploaded,1)
        self.assertTrue(any(f['value']=='Unrecognized resume text' for f in env.all_fields()))
        driver.kind('cancel');driver.button('next');env.max_steps=3000
        driver.complete();self.assertEqual(env.completed,1)

    def test_unknown_required_facts_require_review_never_fabrication(self):
        env=SimulatedWorkdayEnvironment(level=2,trace=True);env.reset(np.random.default_rng(3))
        field=env.current_fields()[0];env._validator.expected[field['id']]=None;field['required']=True
        driver=FixtureDriver(env);driver.kind('focus',target=field['id']);_,reward,done,info=driver.kind('review',target=field['id'])
        self.assertTrue(done);self.assertEqual(env.completed,0)
        self.assertEqual(env.status,'manual_review_missing_information');self.assertLess(reward,2)

    def test_account_and_pending_application_persist_separately_from_weights(self):
        env,_=known_full_fixture(level=4);env.state.accounts.clear();env.max_steps=3000
        driver=FixtureDriver(env);driver.auth(register=True)
        snapshot=env.snapshot();public=env.public_view()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.json';env.state.drafts['pending']=snapshot;env.state.save(path)
            restored=SimulationState.load(path)
            other=SimulatedWorkdayEnvironment(state=restored,vault=env.vault,level=4)
            other.reset(np.random.default_rng(100));other.restore(restored.drafts['pending'])
            self.assertEqual(public,other.public_view());self.assertEqual(env.paid,other.paid)
            self.assertTrue(restored.login(env.tenant,env.email,env.vault.resolve(env.tenant,'password_A')))
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_missing_resume_world_and_mismatched_vault_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                SimulationState.load(Path(directory)/'missing.json',required=True)
        world=SimulationState();world.bind_vault(SecretVault(b'a'*32))
        with self.assertRaises(ValueError):world.bind_vault(SecretVault(b'b'*32))

    def test_vault_persistence_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'vault.json';a=SecretVault.open(path);b=SecretVault.open(path)
            self.assertEqual(a.resolve('tenant','password_A'),b.resolve('tenant','password_A'))
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_no_inbox_or_verification_actions_and_workflows_are_skipped(self):
        env,_=known_full_fixture();env.scenario['verification_required']=True;env.status='skipped_email_verification'
        choices=env.available_actions();self.assertEqual(len(choices),1)
        _,reward,done,_=env.step(0,np.random.default_rng(1))
        self.assertTrue(done);self.assertEqual(env.status,'skipped_email_verification');self.assertEqual(env.completed,0);self.assertLessEqual(reward,0)
        self.assertFalse(any(a.kind in ('inbox','verify') for a in choices))

    def test_seeded_prior_submission_has_a_tenant_account(self):
        profile=SimulatedWorkdayEnvironment().profile
        duplicate_count=0
        for seed in range(100):
            scenario=generate(profile,np.random.default_rng(seed),'test',9)
            if scenario['seed_duplicate']:
                duplicate_count+=1
                self.assertTrue(scenario['seed_existing'])
        self.assertGreater(duplicate_count,0)

    def test_disjoint_label_namespaces_and_reproducible_forms(self):
        profile=SimulatedWorkdayEnvironment().profile
        sets={}
        for split in ('train','validation','test'):
            labels=set()
            for seed in range(20):
                scenario=generate(profile,np.random.default_rng(seed),split,9)
                same=generate(profile,np.random.default_rng(seed),split,9)
                self.assertEqual(fingerprint(scenario),fingerprint(same))
                for name,group in scenario['sections'].items():
                    fields=[f for entry in group for f in entry] if name in ('experience','education') else group
                    labels.update(f['label'] for f in fields)
            sets[split]=labels
        self.assertFalse(sets['train']&sets['test']);self.assertFalse(sets['train']&sets['validation'])

    def test_curriculum_requires_success_and_retains_rehearsal(self):
        c=Curriculum()
        for i in range(30):c.consider({'completed':0.,'field_accuracy':1.,'episodes':64},10000,i)
        self.assertEqual(c.level,1)
        c.consider({'completed':1.,'field_accuracy':1.,'episodes':16},10000,31);self.assertEqual(c.level,1)
        for i in (32,33):c.consider({'completed':.8,'field_accuracy':.9,'episodes':32},10000,i)
        self.assertEqual(c.level,2)
        samples=[c.sample(np.random.default_rng(i)) for i in range(100)]
        self.assertIn(1,samples);self.assertIn(2,samples)


if __name__=='__main__':unittest.main()
