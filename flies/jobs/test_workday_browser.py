"""Opt-in local Chromium contract/physics checks. Scripted fixtures are NOT training."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from flies.jobs.workday import SimulatedWorkdayEnvironment
from flies.jobs.test_workday import FixtureDriver, known_full_fixture


@unittest.skipUnless(os.environ.get('PLAYWRIGHT_BROWSERS_PATH'),'set PLAYWRIGHT_BROWSERS_PATH for Chromium tests')
class WorkdayBrowserTests(unittest.TestCase):
    def setUp(self):
        from playwright.sync_api import sync_playwright
        from flies.jobs.browser import BrowserMirror
        self.driver=sync_playwright().start()
        self.browser=self.driver.chromium.launch(headless=True)
        self.page=self.browser.new_page()
        self.mirror=BrowserMirror(self.page)

    def tearDown(self):
        self.browser.close();self.driver.stop()

    def test_browser_reads_dom_and_executes_exact_rejected_values(self):
        env=SimulatedWorkdayEnvironment(level=1,trace=True);env.reset(np.random.default_rng(3))
        self.mirror.open(env)
        np.testing.assert_array_equal(env.obs(),self.mirror.observation())
        driver=FixtureDriver(env,self.mirror)
        field=env.current_fields()[0]
        driver.kind('focus',target=field['id'])
        wrong=next(a.value for a in env.available_actions() if a.kind=='enter' and a.value!=env._validator.expected[field['id']])
        driver.kind('enter',value=wrong)
        self.assertEqual(self.page.locator('#'+field['id']).input_value(),wrong)
        before=self.mirror.observation()
        self.page.locator('#palette button').first.evaluate('(b)=>b.textContent="changed DOM action label"')
        self.assertFalse(np.array_equal(before,self.mirror.observation()))
        self.assertFalse(self.page.evaluate('Object.keys(window.publicMeta).includes("expected")'))
        self.assertEqual(env.completed,0)

    def test_browser_auth_and_upload_controls_use_selected_values(self):
        env,_=known_full_fixture(level=5);env.max_steps=3000
        self.mirror.open(env);driver=FixtureDriver(env,self.mirror)
        driver.auth(register=not bool(env.state.account(env.tenant,env.email)))
        self.assertTrue(env.authenticated)
        self.assertEqual(env.stage,'resume')
        driver.kind('focus',target='resume_upload');driver.kind('upload',ref='other_document')
        self.assertEqual(env.uploaded,0)
        self.assertEqual(self.page.locator('#resume_upload').evaluate('(n)=>n.files[0].name'),'README.md')
        driver.kind('upload',ref='resume_pdf')
        self.assertEqual(self.page.locator('#resume_upload').evaluate('(n)=>n.files[0].name'),'Elton_Inglewood_Fly_Resume.pdf')
        self.assertEqual(env.uploaded,1)
        payload=str(self.mirror.observation_view())+str(env.traces)
        self.assertNotIn(env.vault.resolve(env.tenant,'password_A'),payload)
        self.assertNotIn(env.vault.resolve(env.tenant,'password_B'),payload)

    def test_secrets_remain_redacted_even_when_selected_for_the_wrong_field(self):
        env,_=known_full_fixture(level=4)
        env._move(env.scenario['stages'].index('auth'))
        self.mirror.open(env);driver=FixtureDriver(env,self.mirror)
        driver.kind('focus',target='auth_email')
        driver.kind('catalog',ref='credentials')
        driver.kind('enter',ref='password_A')
        secret=env.vault.resolve(env.tenant,'password_A')
        self.assertFalse(secret in str(self.mirror.observation_view()))
        self.assertFalse(secret in str(env.traces))
        np.testing.assert_array_equal(env.obs(),self.mirror.observation())

    def test_real_radio_controls_execute_literal_option_selections(self):
        env,_=known_full_fixture(level=9)
        env._move(env.scenario['stages'].index('questions'))
        field=env.current_fields()[0]
        field['kind']='radio';field['options']=['Yes','No'];field['value']=''
        self.mirror.open(env);driver=FixtureDriver(env,self.mirror)
        driver.kind('focus',target=field['id']);driver.kind('choose',value='Yes')
        self.assertTrue(self.page.locator('#'+field['id']).get_by_role('radio',name='Yes',exact=True).is_checked())
        self.assertEqual(field['value'],'Yes')

    def test_station_consumes_same_event_and_does_not_serve_secrets(self):
        from flies.jobs.station_server import StationServer
        env=SimulatedWorkdayEnvironment(level=1,trace=True);env.reset(np.random.default_rng(3))
        with StationServer() as server:
            self.mirror.station_origin=server.origin
            self.mirror.open(env)
            driver=FixtureDriver(env,self.mirror)
            driver.kind('focus',target=env.current_fields()[0]['id'])
            self.assertGreaterEqual(self.page.evaluate('window.stationAck'),1)
            self.assertEqual(self.page.evaluate('window.lastExecutionEvent.policy'),'TEST_FIXTURE_NOT_LEARNING')
            frame=self.page.frame_locator('#execution-station')
            self.assertIn('TEST_FIXTURE_NOT_LEARNING',frame.locator('#hud').inner_text())
            # Asset server is explicitly allowlisted, not a repository file server.
            import urllib.request,urllib.error
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(server.origin+'/flies/jobs/local_state/simulation_vault.json')


if __name__=='__main__':unittest.main()
