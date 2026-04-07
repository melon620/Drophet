# -*- coding: utf-8 -*-
"""
DDI Tox-Predict Project
Script 006: Calculates AE Distributions (Ultimate Nuclear Fix)
Retrieves numeric stats by cross-referencing processed labels and raw data.
Supports both ClinicalTrials.gov API v1 and v2 structures.
"""

import json
import os
from collections import defaultdict

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 錯誤: 找不到檔案 {filepath}!")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"❌ 錯誤: 解析 {filepath} 失敗")
            return None

def get_nctid(trial):
    """
    通用 NCTId 提取器，支援 v1/v2 各種層級。
    """
    if not isinstance(trial, dict): return None
    # 嘗試官方 v2 路徑
    nctid = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
    # 嘗試頂層或常用 Key
    if not nctid:
        nctid = trial.get('nctId') or trial.get('NCTId') or trial.get('id')
    return nctid.strip() if nctid else None

def try_float(val):
    """
    強效類型轉換：處理字串、百分比、逗號。
    """
    try:
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str):
            clean_val = val.replace('%', '').replace(',', '').strip()
            return float(clean_val)
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def find_stats_pairs_anywhere(obj, current_group='Unknown'):
    """
    全域核能搜尋：遞歸掃描所有字典，尋找任何成對的 Affected/Risk 數據。
    """
    results = []
    
    # 定義所有已知變體 (全小寫比對)
    aff_keys = {'numaffected', 'subjectsaffected', 'affected', 'count', 'n_affected', 'seriousnumaffected', 'othernumaffected', 'numseriousadverseevents'}
    risk_keys = {'numatrisk', 'subjectsatrisk', 'total', 'atrisk', 'n_at_risk', 'seriousnumatrisk', 'othernumatrisk', 'denom', 'numsubjectsatrisk'}

    if isinstance(obj, dict):
        g_id = obj.get('groupId', obj.get('id', current_group))
        lower_keys = {k.lower(): k for k in obj.keys()}
        
        hit_aff = next((lower_keys[v] for v in aff_keys if v in lower_keys), None)
        hit_risk = next((lower_keys[v] for v in risk_keys if v in lower_keys), None)
        
        if hit_aff is not None and hit_risk is not None:
            aff_val = try_float(obj[hit_aff])
            risk_val = try_float(obj[hit_risk])
            if risk_val > 0:
                results.append({'groupId': g_id, 'numAffected': aff_val, 'numAtRisk': risk_val})
        
        for v in obj.values():
            results.extend(find_stats_pairs_anywhere(v, g_id))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_stats_pairs_anywhere(item, current_group))
    return results

def main():
    print("--- Phase 1: Ultimate Nuclear AE Distribution Calculation ---")
    
    processed_file = 'updated_clinical_trials.json'
    raw_file = 'special-trials.json'
    
    processed_data = load_json(processed_file)
    raw_data = load_json(raw_file)
    
    if not processed_data or not raw_data:
        print("🚨 關鍵錯誤: 缺少輸入檔案。")
        return

    # 1. 索引原始數據
    raw_map = {get_nctid(t): t for t in raw_data if get_nctid(t)}
    print(f"🧠 已索引 {len(raw_map)} 個原始試驗數據。")

    final_output = []
    total_stats_found = 0
    
    # 2. 跨文件對接處理
    for trial in processed_data:
        nct_id = get_nctid(trial)
        ae_categories = trial.get('AE_Categories', [])
        
        if not nct_id or not ae_categories:
            continue

        raw_record = raw_map.get(nct_id)
        if not raw_record:
            continue

        # 3. 複合提取策略
        found_stats = []
        
        # 策略 A: 直接鎖定 v2 摘要層級 (eventGroups)
        ae_module = raw_record.get('resultsSection', {}).get('adverseEventsModule', {})
        if not ae_module:
            ae_module = raw_record.get('adverseEventsModule', raw_record)

        for eg in ae_module.get('eventGroups', []):
            gid = eg.get('id', 'Unknown')
            # Serious
            s_aff = eg.get('seriousNumAffected', eg.get('numSeriousAdverseEvents'))
            s_risk = eg.get('seriousNumAtRisk', eg.get('numSubjectsAtRisk'))
            if s_aff is not None and s_risk is not None and try_float(s_risk) > 0:
                found_stats.append({'groupId': gid, 'numAffected': try_float(s_aff), 'numAtRisk': try_float(s_risk)})
            # Other
            o_aff = eg.get('otherNumAffected')
            o_risk = eg.get('otherNumAtRisk', s_risk)
            if o_aff is not None and o_risk is not None and try_float(o_risk) > 0:
                found_stats.append({'groupId': gid, 'numAffected': try_float(o_aff), 'numAtRisk': try_float(o_risk)})

        # 策略 B: 掃描具體事件列表 (seriousEvents / otherEvents)
        for e_type in ['seriousEvents', 'otherEvents']:
            for ev in ae_module.get(e_type, []):
                # 檢查 stats 陣列
                for s in ev.get('stats', []):
                    s_res = find_stats_pairs_anywhere(s, ev.get('groupId', 'Unknown'))
                    found_stats.extend(s_res)
                # 檢查事件本身
                e_res = find_stats_pairs_anywhere(ev)
                found_stats.extend(e_res)

        # 策略 C: 全域核能掃描 (最後手段)
        if not found_stats:
            found_stats = find_stats_pairs_anywhere(raw_record)
            
        total_stats_found += len(found_stats)

        # 4. 數據映射至 Matrix 格式
        serious_dist = defaultdict(lambda: defaultdict(float))
        for s in found_stats:
            g_id = s['groupId']
            percentage = (s['numAffected'] / s['numAtRisk']) * 100
            for cat in ae_categories:
                # 每個 Group 保留該類別的最大發生率
                serious_dist[g_id][cat] = max(serious_dist[g_id][cat], percentage)

        final_output.append({
            "nctId": nct_id,
            "seriousEvents": {gid: dict(cats) for gid, cats in serious_dist.items()},
            "otherEvents": {}
        })

    # 5. 儲存結果
    output_file = 'trial_adverse_event_distributions_with_data_separated.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"\n✅ 成功: 處理了 {len(final_output)} 個試驗。")
    print(f"📊 總共搵到 {total_stats_found} 組統計數據。")
    
    if total_stats_found == 0:
        print("🚨 嚴重警告: 依然搵唔到任何數字。")
        print("💡 檢查建議: 請手動打開 special-trials.json。搜尋 'seriousNumAffected'。")
        print("如果搜尋結果係 0，代表你份原始檔已經被整壞咗（冇晒數），你需要重新下載原始數據。")

    print(f"📁 輸出檔案: {output_file}")

if __name__ == "__main__":
    main()