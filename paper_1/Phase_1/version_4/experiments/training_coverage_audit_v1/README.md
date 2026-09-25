# Fixed Training Coverage Audit

No training launched. Existing artifacts are read-only.

| Split | Matches | Full targets | Selected | Target coverage | F80 input coverage |
|---|---:|---:|---:|---:|---:|
| train | 266 | 449025 | 34048 | 7.583% | 99.628% |
| validation | 57 | 96891 | 7296 | 7.530% | 99.599% |

Input coverage counts unique events appearing in at least one F80 history,
divided by events that can precede a target (all except the final event per match).
This does not imply that every possible anchor/context/target combination is trained.

## Event Supervision

| Split | Event | Full targets | Selected | Coverage |
|---|---|---:|---:|---:|
| train | Duel | 123561 | 9242 | 7.48% |
| train | Foul | 5752 | 437 | 7.60% |
| train | Free Kick | 25634 | 1971 | 7.69% |
| train | Goalkeeper leaving line | 899 | 67 | 7.45% |
| train | Interruption | 19221 | 1502 | 7.81% |
| train | Offside | 1093 | 91 | 8.33% |
| train | Others on the ball | 35126 | 2643 | 7.52% |
| train | Pass | 229414 | 17470 | 7.62% |
| train | Save attempt | 2329 | 153 | 6.57% |
| train | Shot | 5996 | 472 | 7.87% |
| validation | Duel | 26800 | 2049 | 7.65% |
| validation | Foul | 1194 | 82 | 6.87% |
| validation | Free Kick | 5546 | 431 | 7.77% |
| validation | Goalkeeper leaving line | 198 | 18 | 9.09% |
| validation | Interruption | 4291 | 326 | 7.60% |
| validation | Offside | 253 | 25 | 9.88% |
| validation | Others on the ball | 7807 | 568 | 7.28% |
| validation | Pass | 49055 | 3657 | 7.45% |
| validation | Save attempt | 521 | 50 | 9.60% |
| validation | Shot | 1226 | 90 | 7.34% |

Player coverage uses positive raw actor IDs, not the candidate-valid loss mask.
See player_coverage.csv for zero/low-supervision players and report.json for provenance.
Three seeds and 24 epochs repeat the same target subset; they do not expand unique target coverage.
