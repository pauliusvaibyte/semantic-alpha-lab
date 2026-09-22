from semantic_alpha.providers.x_stream import posts_from_event, symbol_from_tag_or_text, symbols_from_tag_or_text


def test_stream_event_parsing_and_symbol():
    event={'event_type':'tweet','rule_tag':'semantic-alpha-SOL','tweets':[{'id':'1','text':'Buying $SOL here'}]}
    xs=posts_from_event(event)
    assert len(xs)==1
    assert symbol_from_tag_or_text(xs[0][0],xs[0][1]['text'],['BTCUSDT','SOLUSDT'])=='SOLUSDT'


def test_text_symbol_fallback():
    assert symbol_from_tag_or_text(None,'Watching $BTC breakout',['BTCUSDT','SOLUSDT'])=='BTCUSDT'


def test_multi_asset_detection_from_one_post():
    symbols=["BTCUSDT","ETHUSDT","SOLUSDT","OPUSDT"]
    found=symbols_from_tag_or_text("sol-feed", "Long $ETH vs $BTC while Solana consolidates", symbols)
    assert found == ["SOLUSDT","BTCUSDT","ETHUSDT"] or found == ["SOLUSDT","ETHUSDT","BTCUSDT"]
    # Do not interpret the common English word 'op' as the OP token without a cashtag/full name.
    assert "OPUSDT" not in symbols_from_tag_or_text(None,"great op from the team",symbols)
    assert "OPUSDT" in symbols_from_tag_or_text(None,"Watching $OP and Optimism",symbols)
