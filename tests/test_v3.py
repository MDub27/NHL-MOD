import copy
import datetime as dt
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock
import numpy as np
from nhl_data import Client, DataError, UTC, collect_season, compact_xg, date_string, history_before, normalize_game, schedule, validate_club
from nhl_moneyline import FeatureState, chronological_features, fit, metrics, score
from nhl_report import combinations, save_snapshot


def raw(state='OFF', date='2026-11-13', gid=1):
    return dict(id=gid, gameDate=date, startTimeUTC=date+'T23:00:00Z', gameState=state, gameType=2,
                homeTeam={'abbrev':'BOS','score':3}, awayTeam={'abbrev':'NYR','score':2})


class DataTests(unittest.TestCase):
    def test_final_validation_rejects_missing_scores_and_ties(self):
        g=raw(); del g['homeTeam']['score']
        with self.assertRaises(DataError): normalize_game(g)
        g=raw(); g['awayTeam']['score']=3
        with self.assertRaises(DataError): normalize_game(g)

    def test_live_scores_are_not_history(self):
        live=normalize_game(raw('LIVE'))
        self.assertNotIn('hg',live)
        self.assertEqual(history_before([live],'2026-11-14'),[])

    def test_date_cutoff_includes_neither_same_day_nor_future(self):
        games=[normalize_game(raw(date=d,gid=i)) for i,d in enumerate(['2026-11-13','2026-11-14','2026-11-15'])]
        self.assertEqual([g['date'] for g in history_before(games,'2026-11-14')],['2026-11-13'])
        # NHL date is authoritative even if puck drop is on the following UTC day.
        games[0]['start_utc']='2026-11-14T03:00:00Z'
        self.assertEqual(len(history_before(games,'2026-11-14')),1)

    def test_date_format_and_invalid_dates(self):
        for d in ['20261114','2026-02-30','today','']:
            with self.assertRaises(ValueError):date_string(d)

    def test_club_partial_and_duplicate_payloads(self):
        for p in [{},{'games':[]},{'games':[raw(),raw()]}]:
            with self.assertRaises(DataError):validate_club(p)

    def test_partial_season_cannot_be_cached_as_complete(self):
        class Fake:
            def get(self, endpoint, validator, ttl):
                payload={'games':[raw()]}
                validator(payload)
                return payload
        with self.assertRaisesRegex(DataError,'Incomplete 2025 club schedule'):
            collect_season(Fake(),2025,'2026-11-14')

    def test_requested_day_and_game_count_required(self):
        for payload in [{'gameWeek':[]},{'gameWeek':[{'date':'2026-11-14','numberOfGames':2,'games':[raw(date='2026-11-14')]}]}]:
            class Fake:
                def get(self,endpoint,validator,ttl):
                    validator(payload)
                    return payload
            with self.assertRaises(DataError):schedule(Fake(),'2026-11-14')

    def test_weekly_schedule_inherits_date_from_parent_day(self):
        g=raw(date='2026-11-14');del g['gameDate']
        class Fake:
            def get(self,endpoint,validator,ttl):
                payload={'gameWeek':[{'date':'2026-11-14','numberOfGames':1,'games':[g]}]}
                validator(payload)
                return payload
        self.assertEqual(schedule(Fake(),'2026-11-14')[0]['date'],'2026-11-14')

    def test_cache_ttl_immutable_and_corrupt_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock=[100.]
            c=Client(tmp,clock=lambda:clock[0]);load=Mock(return_value={'ok':1})
            validate=lambda p: self.assertEqual(p,{'ok':1})
            c.cached('key',load,validate,10);c.cached('key',load,validate,10)
            self.assertEqual(load.call_count,1)
            clock[0]=111.;c.cached('key',load,validate,10)
            self.assertEqual(load.call_count,2)
            clock[0]=999.;c.cached('key',load,validate,None)
            self.assertEqual(load.call_count,2)
            (pathlib.Path(tmp)/'key.json').write_text('corrupt')
            c.cached('key',load,validate,10)
            self.assertEqual(load.call_count,3)

    def test_invalid_refresh_does_not_poison_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Client(tmp,clock=lambda:10.)
            def validate(p):
                if not p.get('ok'):raise DataError('partial')
            c.cached('key',lambda:{'ok':1},validate,1)
            path=pathlib.Path(tmp)/'key.json';before=path.read_bytes()
            c.clock=lambda:20.
            with self.assertRaises(DataError):c.cached('key',lambda:{},validate,1)
            self.assertEqual(path.read_bytes(),before)

    def test_retry_timeout_and_permanent_failure(self):
        response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=b'{}'
        opener=Mock(side_effect=[TimeoutError('timeout'),response]);sleep=Mock()
        c=Client(opener=opener,sleeper=sleep,timeout=7)
        self.assertEqual(c.download('https://example.test'),b'{}')
        self.assertEqual(opener.call_count,2);self.assertEqual(opener.call_args.kwargs['timeout'],7)
        opener=Mock(side_effect=urllib.error.HTTPError('x',403,'Forbidden',None,None))
        with self.assertRaises(DataError):Client(opener=opener,sleeper=sleep).download('https://example.test')
        self.assertEqual(opener.call_count,1)
        opener=Mock(side_effect=TimeoutError('timeout'))
        with self.assertRaises(DataError):Client(opener=opener,sleeper=sleep).download('https://example.test')
        self.assertEqual(opener.call_count,3)

    def test_money_puck_filter_and_invalid_values(self):
        header='season,situation,gameId,team,gameDate,xGoalsFor,xGoalsAgainst\n'
        rows='2023,all,2023020001,BOS,20231010,2.5,1.8\n2023,5on5,2023020001,BOS,20231010,1,1\n2023,all,2023030001,BOS,20231010,2,1\n'
        self.assertEqual(len(compact_xg(io.StringIO(header+rows))),1)
        with self.assertRaises(DataError):compact_xg(io.StringIO(header+'2023,all,2023020001,BOS,20231010,nan,1\n'))


class ModelTests(unittest.TestCase):
    def games(self):
        return [dict(id=str(i),date=f'2023-10-{i//2+1:02}',home='BOS' if i%2 else 'NYR',away='NYR' if i%2 else 'BOS',hg=4,ag=2,state='OFF') for i in range(20)]

    def test_same_day_and_future_outcomes_cannot_change_features(self):
        games=self.games();a,_=chronological_features(games,{},min_history=0)
        changed=copy.deepcopy(games);changed[-1]['hg']=100
        b,_=chronological_features(changed,{},min_history=0)
        np.testing.assert_equal(a[-2]['x'],b[-2]['x'])
        np.testing.assert_equal(a[-1]['x'],b[-1]['x'])
        # Today's outcome can change labels but not today's features, including Elo.
        self.assertEqual(a[-1]['elo'],b[-1]['elo'])

    def test_xg_only_enters_history_after_its_game(self):
        games=self.games(); xg={(g['id'],t):{'date':g['date'],'xf':2.,'xa':1.} for g in games for t in ['BOS','NYR']}
        a,_=chronological_features(games,xg,min_history=0)
        xg[('19','BOS')]['xf']=1000
        b,_=chronological_features(games,xg,min_history=0)
        np.testing.assert_equal(a[-1]['x'],b[-1]['x'])

    def test_win_label_includes_one_goal_overtime_or_shootout_wins(self):
        games=self.games();games[0].update(hg=3,ag=2);games[1].update(hg=2,ag=3)
        rows,_=chronological_features(games,{},min_history=0)
        self.assertEqual([r['y'] for r in rows[:2]],[1,0])

    def test_rest_and_missing_xg_flags(self):
        state=FeatureState();g=self.games()[0];state.add(g,{})
        x,q,_=state.features({**g,'date':'2023-10-02'})
        self.assertEqual(x[-1],1);self.assertTrue(np.isnan(x[3]));self.assertEqual(q['home_xg_games'],0)

    def test_holdout_labels_do_not_affect_fit_or_calibration(self):
        rows=[]
        for year in [2023,2024,2025]:
            for i in range(210):
                rows.append({'id':f'{year}-{i}','date':f'{year}-10-01','x':[float(i%3)]*9,'y':i%2,'elo':.54})
        a=fit(rows,'2024-07-01','2025-07-01')
        changed=[{**r,'y':1-r['y']} if r['date']>'2025' else r for r in rows]
        b=fit(changed,'2024-07-01','2025-07-01')
        np.testing.assert_allclose(score(a,rows[-10:])['moneyline_calibrated'],score(b,rows[-10:])['moneyline_calibrated'])

    def test_metrics_and_probability_complements(self):
        m=metrics([1,0],[.9,.1]);self.assertAlmostEqual(m['brier'],.01)
        self.assertAlmostEqual(m['log_loss'],-np.log(.9))
        self.assertEqual(sum(b['games'] for b in m['calibration_bins']),2)

    def test_correlation_bounds_not_independent_same_game(self):
        legs=[{'game_id':'1','selection':'over','probability':.8},{'game_id':'1','selection':'home','probability':.6},{'game_id':'2','selection':'away','probability':.7}]
        combos=combinations(legs)
        self.assertIsNone(combos[0]['estimated_probability'])
        self.assertAlmostEqual(combos[0]['joint_lower_bound'],.4)
        self.assertAlmostEqual(combos[1]['estimated_probability'],.56)
        self.assertTrue(all(r['expected_value'] is None for r in combos))

    def test_snapshots_do_not_overwrite(self):
        report={'date':'2026-11-14','generated_at_utc':'2026-11-14T18:00:00+00:00','games':[]}
        with tempfile.TemporaryDirectory() as tmp:
            p=save_snapshot(report,tmp)
            self.assertTrue((p/'pregame.json').exists())
            with self.assertRaises(FileExistsError):save_snapshot(report,tmp)


if __name__=='__main__':unittest.main()
