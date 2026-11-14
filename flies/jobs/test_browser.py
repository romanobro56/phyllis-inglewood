"""Optional Chromium integration check: set PLAYWRIGHT_BROWSERS_PATH to run."""
import os
import unittest
import numpy as np
from flies.jobs.task import JobsFly


@unittest.skipUnless(os.environ.get('PLAYWRIGHT_BROWSERS_PATH'), 'Chromium integration is opt-in')
class BrowserTests(unittest.TestCase):
    def test_dom_sensors_and_selected_actions_have_no_answer_repair(self):
        from playwright.sync_api import sync_playwright
        from flies.jobs.browser import BrowserMirror
        rng = np.random.default_rng(17)
        env = JobsFly(difficulty=.1)
        env.reset(rng)
        with sync_playwright() as driver:
            browser = driver.chromium.launch(headless=True)
            try:
                mirror = BrowserMirror(browser.new_page())
                mirror.open(env)
                before = mirror.observation(env)
                np.testing.assert_array_equal(before, env.obs())
                env._validator.expected = {env.current['id']: 'unreachable answer'}
                np.testing.assert_array_equal(before, mirror.observation(env))
                mirror.page.locator('label').evaluate('(n)=>n.textContent="Completely unfamiliar prompt"')
                self.assertFalse(np.array_equal(before, mirror.observation(env)))
                # Action zero is executed literally, even though the oracle rejects it.
                selected = env.candidates()[0]['value']
                mirror.execute(env, 0)
                _, reward, _, _ = env.step(0, rng)
                self.assertLess(reward, 0)
                actual = mirror.check(env)
                self.assertEqual(actual[env.current['id']], selected)
                self.assertEqual(mirror.page.locator('body').get_attribute('data-attempted'), None)
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
