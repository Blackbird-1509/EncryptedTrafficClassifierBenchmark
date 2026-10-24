"""Build flows.npz from a folder of pcaps (e.g. ISCXVPN2016), or generate synthetic flows for a smoke test.

    python prepare_data.py --pcap-dir /data/ISCXVPN2016 --out flows.npz --workers 8 --cache-dir cache
    python prepare_data.py --synthetic --out flows_synth.npz
"""
import argparse

import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="flows.npz")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-flows-per-file", type=int, default=None,
                    help="randomly keep at most this many flows per pcap (caps huge p2p captures)")
    ap.add_argument("--max-packets-per-file", type=int, default=None,
                    help="stop reading each pcap after this many TCP/UDP packets (fast but only sees the start of each capture)")
    ap.add_argument("--cache-dir", default=None,
                    help="cache parsed results per pcap here; re-runs and added folders then skip finished files")
    ap.add_argument("--parser", choices=["auto", "scapy"], default="auto",
                    help="auto = fast dpkt parser (falls back to Scapy if needed)")
    a = ap.parse_args()

    if a.synthetic:
        sizes, ts, y, groups = D.make_synthetic()
    elif a.pcap_dir:
        sizes, ts, y, groups = D.build_from_pcap_dir(a.pcap_dir, a.workers, a.max_flows_per_file,
                                                    a.max_packets_per_file, a.cache_dir, a.parser)
    else:
        ap.error("give --pcap-dir or --synthetic")

    D.save(a.out, sizes, ts, y, groups)
    print(f"\nsaved {a.out}: {len(y)} flows")
    for i, c in enumerate(D.CLASSES):
        print(f"  {c:14s} {(y == i).sum():7d}")


if __name__ == "__main__":   # required on Windows/macOS: worker processes re-import this file
    main()
