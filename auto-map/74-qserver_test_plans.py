import os
import json
import tifffile as tf
#from hxntools.scan_info import get_scan_positions

def _get_flyscan_dimensions(hdr):
    start_doc = hdr.start
    # 2D_FLY_PANDA: prefer 'dimensions', fallback to 'shape'
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        if 'dimensions' in start_doc:
            return start_doc['dimensions']
        elif 'shape' in start_doc:
            return start_doc['shape']
        else:
            raise ValueError("No dimensions or shape found for 2D_FLY_PANDA scan")
    # rel_scan: use 'shape' or 'num_points'
    elif start_doc.get('plan_name') == 'rel_scan':
        if 'shape' in start_doc:
            return start_doc['shape']
        elif 'num_points' in start_doc:
            return [start_doc['num_points']]
        else:
            raise ValueError("No shape or num_points found for rel_scan")
    else:
        raise ValueError("Unknown scan type for _get_flyscan_dimensions")

def export_xrf_roi_data(scan_id, norm = 'sclr1_ch4', elem_list = [], wd = '.'):

    hdr = db[int(scan_id)]
    scan_id = hdr.start["scan_id"]
   
    channels = [1, 2, 3]
    #print(f"{elem_list = }")
    print(f"[DATA] fetching XRF ROIs")
    scan_dim = _get_flyscan_dimensions(hdr)
    print(f"[DATA] fetching scalar values")

    scalar = np.array(list(hdr.data(norm))).squeeze()
    print(f"[DATA] fetching scalar {norm} values done")

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        if norm !=None:
            spectrum = spectrum/scalar
        xrf_img = spectrum.reshape(scan_dim)
        tf.imwrite(os.path.join(wd,f"scan_{scan_id}_{elem}.tiff"), xrf_img)


def export_scan_params(sid=-1, zp_flag=True, save_to=None):
    """
    Fetch scan parameters, ROI positions, step size, and the full start_doc
    for scan `sid`.  Optionally write them out as JSON.

    Returns a dict with:
      - scan_id
      - start_doc
      - roi_positions
      - step_size (computed from scan_input for 2D_FLY_PANDA)
    """
    # 1) Pull the header
    hdr = db[int(sid)]
    start_doc = dict(hdr.start)  # cast to plain dict

    # 2) Grab the baseline table and build the ROI dict
    tbl = db.get_table(hdr, stream_name='baseline')
    row = tbl.iloc[0]
    if zp_flag:
        roi = {
            "zpssx":    float(row["zpssx"]),
            "zpssy":    float(row["zpssy"]),
            "zpssz":    float(row["zpssz"]),
            "smarx":    float(row["smarx"]),
            "smary":    float(row["smary"]),
            "smarz":    float(row["smarz"]),
            "zp.zpz1":  float(row["zpz1"]),
            "zpsth":    float(row["zpsth"]),
            "zps.zpsx": float(row["zpsx"]),
            "zps.zpsz": float(row["zpsz"]),
        }
    else:
        roi = {
            "dssx":  float(row["dssx"]),
            "dssy":  float(row["dssy"]),
            "dssz":  float(row["dssz"]),
            "dsx":   float(row["dsx"]),
            "dsy":   float(row["dsy"]),
            "dsz":   float(row["dsz"]),
            "sbz":   float(row["sbz"]),
            "dsth":  float(row["dsth"]),
        }

    # 3) Compute unified step_size from scan_input
    scan_info = start_doc.get("scan", {})
    si = scan_info.get("scan_input", [])
    if scan_info.get("type") == "2D_FLY_PANDA" and len(si) >= 3:
        fast_start, fast_end, fast_N = si[0], si[1], si[2]
        step_size = abs(fast_end - fast_start) / fast_N
    else:
        raise ValueError(f"Cannot compute step_size for scan type {scan_info.get('type')}")

    # 4) Assemble the result dict
    result = {
        "scan_id":       int(sid),
        "start_doc":     start_doc,
        "roi_positions": roi,
        "step_size":     float(step_size),
    }

    # 5) Optionally write out JSON
    if save_to:
        if os.path.isdir(save_to):
            filename = os.path.join(save_to, f"scan_{sid}_params.json")
        else:
            filename = save_to if save_to.lower().endswith(".json") else save_to + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

    return result

def fly2d_qserver_scan_export(label,
                           dets,
                           mot1, mot1_s, mot1_e, mot1_n,
                           mot2, mot2_s, mot2_e, mot2_n,
                           exp_t,
                           roi_positions=None,
                           scan_id=None,
                           zp_move_flag=1,
                           smar_move_flag=1,
                           ic1_count=55000,
                           # **POST-SCAN EXPORTS**
                           elem_list=None,           # list of elements for XRF
                           export_norm='sclr1_ch4',  # channel to normalize by
                           data_wd='.',              # where to write TIFFs
                           pos_save_to=None):        # JSON filename or dir
    """
    1) Optionally recover a previous scan or ROI dict
    2) Do beam/flux checks
    3) Run fly2dpd
    4) Export XRF-ROI data TIFFs
    5) Save final ROI positions JSON
    """
    print(f"{label} starting…")
    RE.md["scan_name"] = str(label)

    # — 1) RECOVERY —
    moved = False
    # If a valid scan_id is provided (truthy), recover from that scan
    if scan_id:
        yield from recover_zp_scan_pos(scan_id,
                                       zp_move_flag=zp_move_flag,
                                       smar_move_flag=smar_move_flag,
                                       move_base=1)
        moved = True

    # Else if ROI positions dict/string provided, and not all values None
    elif roi_positions:
        if isinstance(roi_positions, str):
            roi_positions = json.loads(roi_positions)
        # Filter out keys with None values
        non_null = {k: v for k, v in roi_positions.items() if v is not None}
        if non_null:
            for key, val in non_null.items():
                if key != "zp.zpz1":
                    yield from bps.mov(eval(key), val)
                else:
                    yield from mov_zpz1(val)
                print(f"  → {key} @ {val:.3f}")
            yield from check_for_beam_dump(threshold=5000)
            if sclr2_ch2.get() < ic1_count * 0.9:
                yield from peak_the_flux()
            moved = True

    if not moved:
        print("[RECOVERY] no ROI recovery requested; skipping motor moves.")

    # — 2) FLY SCAN —
    yield from fly2dpd(dets,
                       mot1, mot1_s, mot1_e, mot1_n,
                       mot2, mot2_s, mot2_e, mot2_n,
                       exp_t)

    # — 3) POST-SCAN EXPORTS —
    hdr = db[-1]
    last_id = hdr.start["scan_id"]
    print(f"[POST] exporting XRF ROI data for scan {last_id}…")
    export_xrf_roi_data(last_id,
                        norm=export_norm,
                        elem_list=elem_list or [],
                        wd=data_wd)

    if pos_save_to:
        print(f"[POST] saving ROI positions JSON to {pos_save_to}…")
        export_scan_params(sid=last_id, zp_flag=True, save_to=pos_save_to)

    print("[POST] done.")


def send_fly2d_to_queue(label,
                        dets,
                        mot1, mot1_s, mot1_e, mot1_n,
                        mot2, mot2_s, mot2_e, mot2_n,
                        exp_t,
                        roi_positions=None,
                        scan_id=None,
                        zp_move_flag=1,
                        smar_move_flag=1,
                        ic1_count = 55000,
                        elem_list=None,
                        export_norm='sclr2_ch4',
                        data_wd='.',
                        pos_save_to=None):
    det_names = [d.name for d in eval(dets)]
    roi_json = ""
    if isinstance(roi_positions, dict):
        roi_json = json.dumps(roi_positions)
    elif isinstance(roi_positions, str):
        roi_json = roi_positions

    RM.item_execute(BPlan("fly2d_qserver_scan_export",
                      label,
                      det_names,
                      mot1, mot1_s, mot1_e, mot1_n,
                      mot2, mot2_s, mot2_e, mot2_n,
                      exp_t,
                      roi_json,
                      scan_id or "",
                      zp_move_flag,
                      smar_move_flag,
                      ic1_count,
                      json.dumps(elem_list or []),
                      export_norm,
                      data_wd,
                      pos_save_to or ""
                      ))
import os
import json
import time
from databroker import db
from bluesky import plans as bps

# Assume these are imported or defined elsewhere in your namespace:
# send_fly2d_to_queue, export_xrf_roi_data, export_scan_params, recover_zp_scan_pos,
# fly2dpd, check_for_beam_dump, peak_the_flux

def wait_for_queue_done(poll_interval=2.0):
    """
    Block until the QServer queue is empty and the manager goes idle.
    """
    print("[WAIT] polling queue status...", end="", flush=True)
    while True:
        st = RM.status()
        if st['items_in_queue'] == 0 and st['manager_state'] == 'idle':
            print(" done.")
            return
        print(".", end="", flush=True)
        time.sleep(poll_interval)


def submit_and_export(**params):
    """
    Enqueue a scan, wait for completion, then export XRF TIFFs and ROI JSON.
    Expects the same params as load_and_queue would unpack.
    """
    # 1) enqueue
    label = params.get('label', '')
    print(f"[SUBMIT] queueing scan '{label}' …")
    send_fly2d_to_queue(**params)

    # 2) wait
    wait_for_queue_done()

    # 3) grab last scan_id
    hdr = db[-1]
    last_id = hdr.start['scan_id']
    print(f"[EXPORT] scan {last_id} finished; exporting data…")

    # 4) export XRF
    export_xrf_roi_data(last_id,
                        norm=params.get('export_norm', 'sclr2_ch4'),
                        elem_list=params.get('elem_list', []),
                        wd=params.get('data_wd', '.'))

    # 5) save ROI
    pos_save_to = params.get('pos_save_to')
    if pos_save_to:
        export_scan_params(sid=last_id,
                           zp_flag=params.get('zp_move_flag', True),
                           save_to=pos_save_to)

    print("[DONE] all exports complete.")


def load_and_queue(json_path, ):
    """
    Load scan parameters from JSON, compute necessary fields,
    and either enqueue only or enqueue+export based on a flag.

    JSON can include an optional 'block_and_export': true to wait and post-process.
    """
    # 1) Read main params
    with open(json_path, 'r') as f:
        params = json.load(f)

    # 2) Extract blocking flag
    block_and_export = params.pop('block_and_export', True)

    # 3) Load ROI from separate file if requested
    roi_file = params.pop('roi_positions_file', None)
    if roi_file:
        if not os.path.isfile(roi_file):
            raise FileNotFoundError(f"ROI file not found: {roi_file}")
        with open(roi_file, 'r') as rf:
            params['roi_positions'] = json.load(rf)
    elif isinstance(params.get('roi_positions'), str) and os.path.isfile(params['roi_positions']):
        with open(params['roi_positions'], 'r') as rf:
            params['roi_positions'] = json.load(rf)

    # 4) Compute mot1_n & mot2_n from a single step_size
    if 'step_size' in params:
        step = params.pop('step_size')
        params['mot1_n'] = int(abs(params['mot1_e'] - params['mot1_s']) / step)
        params['mot2_n'] = int(abs(params['mot2_e'] - params['mot2_s']) / step)

    # 5) Ensure dets is a string literal for eval()
    if isinstance(params.get('dets'), list):
        params['dets'] = repr(params['dets'])

    # 6) Dispatch
    if block_and_export:
        submit_and_export(**params)
    else:
        send_fly2d_to_queue(**params)


'''

In [7]: RM.status()
Out[7]: 
{'msg': 'RE Manager v0.0.21',
 'items_in_queue': 0,
 'items_in_history': 0,
 'running_item_uid': 'ed947801-8463-4948-9f1e-6593eddd5045',
 'manager_state': 'executing_queue',
 'queue_stop_pending': False,
 'queue_autostart_enabled': False,
 'worker_environment_exists': True,
 'worker_environment_state': 'executing_plan',
 'worker_background_tasks': 0,
 're_state': 'running',
 'ip_kernel_state': 'busy',
 'ip_kernel_captured': True,
 'pause_pending': False,
 'run_list_uid': '21bfb9cf-e6a2-4168-995f-4e772d88c6e5',
 'plan_queue_uid': 'fbfac899-bbaf-422b-9a37-d3fc39515ade',
 'plan_history_uid': 'd824d0d8-ac5a-4b72-a16c-46a28b8d7b6d',
 'devices_existing_uid': 'ea9dff2e-f70d-4367-9c24-e4659d32da6c',
 'plans_existing_uid': 'fcd143b9-4b51-41ef-9e6b-f6800c119956',
 'devices_allowed_uid': '63cfd82b-98df-4eab-a10e-0b15dd6789c0',
 'plans_allowed_uid': '73f190db-31b3-407c-917f-23fda16d4086',
 'plan_queue_mode': {'loop': False, 'ignore_failures': False},
 'task_results_uid': 'db51a825-2a36-4a7b-ae92-47f22b497703',
 'lock_info_uid': '9a2348d0-a2e7-4155-a8d5-9fea5329f7e4',
 'lock': {'environment': False, 'queue': False}}

In [8]: RM.status()
Out[8]: 
{'msg': 'RE Manager v0.0.21',
 'items_in_queue': 0,
 'items_in_history': 1,
 'running_item_uid': None,
 'manager_state': 'idle',
 'queue_stop_pending': False,
 'queue_autostart_enabled': False,
 'worker_environment_exists': True,
 'worker_environment_state': 'idle',
 'worker_background_tasks': 0,
 're_state': 'idle',
 'ip_kernel_state': 'idle',
 'ip_kernel_captured': False,
 'pause_pending': False,
 'run_list_uid': '24e164de-b085-4100-8e0c-351d469e1610',
 'plan_queue_uid': '3d5711fc-6232-4ed4-a104-2c976a840f20',
 'plan_history_uid': '53139750-0762-402c-a88c-e9e76de1d9e4',
 'devices_existing_uid': 'ea9dff2e-f70d-4367-9c24-e4659d32da6c',
 'plans_existing_uid': 'fcd143b9-4b51-41ef-9e6b-f6800c119956',
 'devices_allowed_uid': '63cfd82b-98df-4eab-a10e-0b15dd6789c0',
 'plans_allowed_uid': '73f190db-31b3-407c-917f-23fda16d4086',
 'plan_queue_mode': {'loop': False, 'ignore_failures': False},
 'task_results_uid': 'db51a825-2a36-4a7b-ae92-47f22b497703',
 'lock_info_uid': '9a2348d0-a2e7-4155-a8d5-9fea5329f7e4',
 'lock': {'environment': False, 'queue': False}}

'''