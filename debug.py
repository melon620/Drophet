import json

def check_data():
    try:
        # 睇最原始嘅 data
        with open('special-trials.json', 'r', encoding='utf-8') as f:
            trials = json.load(f)
            
        print(f"Total trials loaded: {len(trials)}")
        
        for trial in trials:
            ae_module = trial.get('resultsSection', {}).get('adverseEventsModule', {})
            serious = ae_module.get('seriousEvents', [])
            other = ae_module.get('otherEvents', [])
            
            # 只要搵到任何一個有紀錄嘅 trial，就印佢個真實結構出嚟睇
            if serious or other:
                print("\n=== 🎯 FOUND AE DATA! Exact JSON Structure of the first event ===")
                if serious:
                    print(json.dumps(serious[0], indent=2))
                else:
                    print(json.dumps(other[0], indent=2))
                return
                
        print("❌ 完全搵唔到任何 seriousEvents 或 otherEvents。")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()