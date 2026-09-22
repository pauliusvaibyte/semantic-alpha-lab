from semantic_alpha.storage import Store


def test_holdout_registry_is_persistent(tmp_path):
    s=Store(str(tmp_path/'h.db'))
    assert s.holdout_opening('abc') is None
    s.save_holdout_opening('abc','15m','proto',{'status':'PASS'},reopened=False)
    r=s.holdout_opening('abc')
    assert r['status']=='PASS' and r['protocol_hash']=='proto' and not r['reopened']
