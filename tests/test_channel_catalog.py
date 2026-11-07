from scripts.collect_channel import catalog


def test_other_game_wins_over_diepio_keyword():
    rows = catalog({'entries': [
        {'id': 'TK6P8CkUMeU', 'title': 'I BECAME A MONSTER Diep.io new rival? - Tanksmith.io'},
        {'id': 'YtomeIYBAhE', 'title': 'My WASD Skills Diep.io'},
        {'id': 'erfWgYS9WJ4', 'title': 'HOW TO GET INSIDE THE MAZE WALL!!!'},
    ]})
    assert [r['selection'] for r in rows] == [
        'excluded_other_game', 'diep_title_candidate', 'ambiguous_review_required']
    assert all(not r['training_approved'] for r in rows)
