# Canonical benchmark prompts (cluster speedup campaign)

Fixed prompt text for every A/B in the cluster tokens/sec campaign. Do not
edit these strings; new prompts get new names. Token counts are approximate
and tokenizer-dependent — the fixed *text* is the contract, reported token
counts come from each run's log.

All cluster runs use the chat path (`master-chat`), greedy decode (temp 0),
64 max tokens unless the receipt says otherwise.

## PROMPT_SHORT (~16 tokens)

```
Explain in one sentence why the sky appears blue during the day.
```

## PROMPT_LONG (~145 tokens)

```
You are helping design a small weather station for a remote farm. The station
must run from a solar panel and a battery, survive cold winters, and report
temperature, humidity, wind speed, and rainfall once every ten minutes over a
low-bandwidth radio link. The farmer cares most about frost warnings for the
orchard, which need to be timely and reliable even when the network is flaky.
Describe a sensible hardware and software architecture for this station.
Explain which sensors you would pick, how you would manage power so the
battery lasts through a week of cloudy days, how you would buffer and resend
readings when the radio link drops, and what simple on-device rule you would
use to raise a frost alarm without waiting for the server.
```
