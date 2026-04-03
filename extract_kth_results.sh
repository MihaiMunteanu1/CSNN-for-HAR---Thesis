#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-csnn-simulator-build/result}"
OUT_CSV="${2:-kth_experiments_summary.csv}"

cat > "$OUT_CSV" <<'CSV'
log_file,experiment_name,random_seed_log,run_start,run_end,duration,train_samples,test_samples,train_videos,test_videos,layer_name,layer_type,epoch,filter_w,filter_h,filter_k,filter_num,wta_infer,sampler,activity_train_sparsity,activity_train_active,activity_test_sparsity,activity_test_active,svm_accuracy,svm_correct,svm_total
CSV

find "$ROOT_DIR" -type f -regextype posix-extended -regex '.*/seed_[0-9]+/log_kth.*\.(txt|log)$' | sort | while read -r LOG; do
  awk -v log_file="$(basename "$LOG")" '
  function trim(s){ gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
  function depercent(s){ gsub("%","",s); return s }
  function count_char(s, c,    t,n){ t=s; n=gsub(c, "", t); return n }

  BEGIN{
    exp_name=""; seed_log=""; run_start=""; run_end=""; duration="";
    train_samples=""; test_samples=""; train_videos=""; test_videos="";
    in_layer=0; brace_depth=0; cur_layer=""; cur_ltype="";
    in_activity=0; act_layer=""; act_phase="";
    svm_layer="";
  }

  {
    line=$0;

    if (line ~ /^Random seed:/) seed_log=trim(substr(line, index(line,":")+1));
    if (line ~ /^Run start at /) run_start=trim(substr(line,13));
    if (line ~ /^Run end at /) run_end=trim(substr(line,11));
    if (line ~ /^Duration:/) duration=trim(substr(line, index(line,":")+1));

    if (line ~ /^Load [0-9]+ train samples from /) {
      split(line, a, " ");
      train_samples=a[2];
      if (match(line, /\[([0-9]+)\]/, m)) train_videos=m[1];
    }
    if (line ~ /^Load [0-9]+ test samples from /) {
      split(line, a, " ");
      test_samples=a[2];
      if (match(line, /\[([0-9]+)\]/, m)) test_videos=m[1];
    }

    if (match(line, /^([A-Za-z0-9_]+)[ \t]+[0-9]+:[ \t]+([A-Za-z0-9_]+)[ \t]+\[/, m)) {
      cur_ltype=m[1]; cur_layer=m[2];
      l_seen[cur_layer]=1;
      if (!(cur_layer in l_type)) l_type[cur_layer]=cur_ltype;
    }

    if (match(line, /^Layer\.([A-Za-z0-9_]+)[ \t]+\(([A-Za-z0-9_]+)\)[ \t]*\{/, m)) {
      in_layer=1;
      brace_depth=1;
      cur_ltype=m[1];
      cur_layer=m[2];
      l_seen[cur_layer]=1;
      l_type[cur_layer]=cur_ltype;
      next;
    }

    if (in_layer) {
      if (match(line, /^[ \t]*epoch:[ \t]*([0-9]+)/, m)) l_epoch[cur_layer]=m[1];
      if (match(line, /^[ \t]*filter_width:[ \t]*([0-9]+)/, m)) l_fw[cur_layer]=m[1];
      if (match(line, /^[ \t]*filter_height:[ \t]*([0-9]+)/, m)) l_fh[cur_layer]=m[1];
      if (match(line, /^[ \t]*filter_conv_depth:[ \t]*([0-9]+)/, m)) l_fk[cur_layer]=m[1];
      if (match(line, /^[ \t]*filter_number:[ \t]*([0-9]+)/, m)) l_fn[cur_layer]=m[1];
      if (match(line, /sampler:[ \t]*Sampler\.([A-Za-z0-9_]+)/, m)) l_sampler[cur_layer]=m[1];

      if (index(line, "wta_infer:") > 0) {
        n = split(line, parts, ":");
        if (n >= 2) {
          v = trim(parts[2]);
          sub(/[ \t].*$/, "", v);
          l_wta[cur_layer] = v;
        }
      }

      tmp1 = line; opens = gsub(/\{/, "", tmp1);
      tmp2 = line; closes = gsub(/\}/, "", tmp2);
      brace_depth += opens - closes;

      if (brace_depth <= 0) {
        in_layer=0;
        brace_depth=0;
      }
    }

    if (match(line, /:[ \t]*([A-Za-z0-9_]+)-([A-Za-z0-9_]+)/, m)) exp_name=m[1];

    if (match(line, /-([A-Za-z0-9_]+),[ \t]*analysis Activity:/, m)) {
      in_activity=1; act_layer=m[1]; act_phase=""; next;
    }

    if (in_activity) {
      if (line ~ /^\* train set:/) { act_phase="train"; next }
      if (line ~ /^\* test set:/)  { act_phase="test"; next }

      if (match(line, /^[ \t]*Sparsity:[ \t]*([0-9.eE+-]+)/, m)) {
        if (act_phase=="train") a_tr_sp[act_layer]=m[1];
        else if (act_phase=="test") a_te_sp[act_layer]=m[1];
      }
      if (match(line, /^[ \t]*Active unit:[ \t]*([0-9.eE+-]+%?)/, m)) {
        v=depercent(m[1]);
        if (act_phase=="train") a_tr_ac[act_layer]=v;
        else if (act_phase=="test") a_te_ac[act_layer]=v;
      }

      if (line ~ /analysis (Coherence|Svm|SaveOutput):/ || line ~ /^===SVM===/) {
        in_activity=0; act_phase="";
      }
    }

    if (match(line, /-([A-Za-z0-9_]+),[ \t]*analysis Svm:/, m)) { svm_layer=m[1]; next }
    if (match(line, /^[ \t]*classification rate:[ \t]*([0-9.]+)%[ \t]*\(([0-9]+)\/([0-9]+)\)/, m)) {
      s_acc[svm_layer]=m[1]; s_ok[svm_layer]=m[2]; s_tot[svm_layer]=m[3];
    }
  }

  END{
    for (layer in l_seen) {
      sampler=(layer in l_sampler)?l_sampler[layer]:"";
      if (sampler=="" && l_type[layer] ~ /HOG_Convolution3D/) sampler="HOGSampler3D";
      if (sampler=="" && l_type[layer] ~ /Convolution3D/) sampler="(not_logged)";
      wta=(layer in l_wta)?l_wta[layer]:"";

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        log_file,exp_name,seed_log,run_start,run_end,duration,
        train_samples,test_samples,train_videos,test_videos,
        layer,l_type[layer],l_epoch[layer],l_fw[layer],l_fh[layer],l_fk[layer],l_fn[layer],
        wta,sampler,
        a_tr_sp[layer],a_tr_ac[layer],a_te_sp[layer],a_te_ac[layer],
        s_acc[layer],s_ok[layer],s_tot[layer];
    }
  }' "$LOG" >> "$OUT_CSV"
done

echo "Done: $OUT_CSV"