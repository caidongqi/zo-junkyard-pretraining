#!/usr/bin/env bash

set -euo pipefail

TARGET_NAME="parallel_sweep.sh"

declare -a JOB_PIDS=()
declare -a DISPLAY_PIDS=()

gather_jobs() {
    local -a raw_pids=()
    mapfile -t raw_pids < <( (pgrep -f "parallel_sweep\.sh" || true) | sort -n )

    JOB_PIDS=()
    for pid in "${raw_pids[@]}"; do
        [[ -z "$pid" ]] && continue
        [[ "$pid" -eq $$ ]] && continue
        if ! ps -p "$pid" > /dev/null 2>&1; then
            continue
        fi
        local comm
        comm=$(ps -p "$pid" -o comm= | tr -d '[:space:]')
        case "$comm" in
            bash|sh|dash|ksh|zsh|parallel_sweep.sh)
                JOB_PIDS+=("$pid")
                ;;
        esac
    done
}

render_jobs() {
    DISPLAY_PIDS=()

    if [ "${#JOB_PIDS[@]}" -eq 0 ]; then
        echo "No running $TARGET_NAME jobs were found."
        return 1
    fi

    echo "Active $TARGET_NAME jobs:"
    echo "-----------------------------------------------"

    local idx=1
    for pid in "${JOB_PIDS[@]}"; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            continue
        fi

        DISPLAY_PIDS+=("$pid")

        local meta
        meta=$(ps -p "$pid" -o pid=,pgid=,ppid=,etime= | sed -e 's/^ *//' -e 's/  */ /g')
        local cmd
        cmd=$(ps -p "$pid" -o cmd= | sed 's/^ *//')

        echo "[$idx] PID/PGID/PPID/ETIME: $meta"
        echo "    CMD: $cmd"

        local children
        children=$(pgrep -P "$pid" || true)
        if [ -n "$children" ]; then
            echo "    Children:"
            for child_pid in $children; do
                [[ -z "$child_pid" ]] && continue
                if ! ps -p "$child_pid" > /dev/null 2>&1; then
                    continue
                fi
                local child_meta
                child_meta=$(ps -p "$child_pid" -o pid=,pgid=,ppid=,etime= | sed -e 's/^ *//' -e 's/  */ /g')
                local child_cmd
                child_cmd=$(ps -p "$child_pid" -o cmd= | sed 's/^ *//')
                echo "      - $child_meta | $child_cmd"
            done
        fi

        echo ""
        idx=$((idx + 1))
    done

    if [ "${#DISPLAY_PIDS[@]}" -eq 0 ]; then
        echo "No running $TARGET_NAME jobs were found."
        return 1
    fi

    return 0
}

kill_tree() {
    local pid="$1"
    local signal="$2"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        return
    fi

    local children
    children=$(pgrep -P "$pid" || true)
    for child_pid in $children; do
        kill_tree "$child_pid" "$signal"
    done

    kill "-$signal" "$pid" 2>/dev/null || true
}

kill_job() {
    local pid="$1"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "PID $pid is no longer running."
        return
    fi

    echo "Sending SIGTERM to PID $pid and its children..."
    kill_tree "$pid" TERM
    sleep 2

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "PID $pid is still running. Sending SIGKILL..."
        kill_tree "$pid" KILL
    fi

    echo "Done handling PID $pid."
}

main_loop() {
    while true; do
        gather_jobs

        if ! render_jobs; then
            read -rp "No jobs to manage. Press Enter to refresh or type q to quit: " choice
            case "$choice" in
                q|Q) exit 0 ;;
                *) continue ;;
            esac
        fi

        read -rp "Select job to kill ([1-${#DISPLAY_PIDS[@]}], e.g. 1 3 4; a=kill all, r=refresh, q=quit): " choice

        case "$choice" in
            q|Q)
                exit 0
                ;;
            r|R|"")
                continue
                ;;
            a|A)
                for pid in "${DISPLAY_PIDS[@]}"; do
                    kill_job "$pid"
                done
                ;;
            *)
                local invalid=false
                local killed_any=false
                for token in $choice; do
                    if [[ "$token" =~ ^[0-9]+$ ]]; then
                        local index=$((token - 1))
                        if [ "$index" -ge 0 ] && [ "$index" -lt "${#DISPLAY_PIDS[@]}" ]; then
                            kill_job "${DISPLAY_PIDS[$index]}"
                            killed_any=true
                        else
                            echo "Invalid selection: $token"
                            invalid=true
                        fi
                    else
                        echo "Unrecognized input token: $token"
                        invalid=true
                    fi
                done
                if [ "$invalid" = false ] && [ "$killed_any" = false ]; then
                    echo "No valid job indices provided."
                fi
                ;;
        esac
    done
}

main_loop


