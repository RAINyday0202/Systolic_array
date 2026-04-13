import re #정규화 표현식 라이브러리 (텍스트 패턴 검색용)
import sys #시스템 관련 라이브러리 (커맨드라인 인자)
import numpy as np #(행렬 연산 라이브러리)

# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

# Matches entries like  c7571d3d(-55069.2383)  or  4224(41.0000)
# Captures the decimal number inside the parentheses.
# re.compile : 텍스트에서 특정 패턴을 찾는 도구, compile은 해당 패턴을 자주 사용하기 위해 준비함
# python에서 \는 명령어로 인식하기 때문에, 앞쪽에 r을 작성하여 문자열로 인식하게 함
# 출력되는 텍스트 앞쪽에는 16진수를 사용하기 때문에 0-9a-fA-F 를 사용하여, 0~9 a~fA~F의 값을 사용함을 명시
# 이후 (Floating point)형태가 나옴. \\로 묶어서 해당 숫자를 캡처하는것을 의미
#앞쪽에는 + 또는 -의 부호가 작성되어 있으며, 부호가 작성되어있지 않을수도 있기 때문에 ? 를 뒤에 사용
#\d+는 1자 이상의 십진수가 작성되어있는 것을 의미
# 이후에는 (\.\d+)로 .xx가 작성되어 있지만, 작성되어 있지 않을 수도 있기 때문에 (~)? 를 작성함.
#하지만, 문법상 괄호 내에 있는것을 캡처하여 저장하기 때문에, 괄호 내에 (?:~)를 작성하여 캡쳐하지 않도록 작성

DECIMAL_RE = re.compile(r"[0-9a-fA-F]+\(([+-]?\d+(?:\.\d+)?)\)")

#parse_row 함수
#.finditer : 이전에 Decimal_RE에서 찾은 것들. 라인별로 나오고, group(0)이 전체숫자, group(1)이 캡쳐한 floating point숫자로 나온다.
#finditer가 찾은 결과를 m 에 저장하고, m에서 group(1)을 리턴
def parse_row(line):
    """Extract all decimal values from one 'RowN: ...' line."""
    return [float(m.group(1)) for m in DECIMAL_RE.finditer(line)]

# parse_matrix 함수
# 라인별로 받으며, 몇번째 줄부터 읽을지, 행렬 사이즈가 몇인지 인자로 받는다.
# 현재 8x8을 사용하므로 size는 8을 받는다.
#text file에서 row에 대한 정보다 2번째 줄부터 나오기 때문에 start_idx도 2로 받는다.
# mat=[] 비어있느 ㄴ인덱스 만들기.
#row 8개를 받기 위해 offset으로 해당 반복문을 8번 반본
# parse_row 명령어로 row별로 값 받아서 row_vals에 저장
# assert : 통과하지 않을시 에러 출력 명령어
# len(row_vals)가 사이즈에 맞지 않으면 에러 발생. size가 8개이기 때문에 요소가 8개 여야 한다.
# amt 행렬에 값 추가
# 이렇게 8x8로 채워진 것을 numpy 행렬로 바꿔서 리턴. 타입은 float 64 사용
def parse_matrix(lines, start_idx, size):
    """
    Read `size` consecutive Row lines starting at lines[start_idx].
    Returns a (size x size) float64 numpy array.
    """
    mat = []
    for offset in range(size):
        row_vals = parse_row(lines[start_idx + offset])
        assert len(row_vals) == size, (
            "Expected {} values on row, got {}: {}".format(
                size, len(row_vals), lines[start_idx + offset].strip()
            )
        )
        mat.append(row_vals)
    return np.array(mat, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Main parser: walks the file and yields (test_id, W, A, C_hw) per test
# ──────────────────────────────────────────────────────────────────────────────
#parse_simulation_output 함수
# 인자로 파일 경로와 사이즈(8)을 받는다.

def parse_simulation_output(filepath, size=8):

    """
    # 한줄씩 리스트로 만들기

    ex. lines = [
    "=== TEST 1 ===\n",       # lines[0]
    "Weight Matrix W\n",       # lines[1]
    "Row0: c7571d3d(...)\n",   # lines[2]
]
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    """
    줄 하나씩 꺼내기.
    line[0] = "=== TEST 1 ===\n"
    
    """
    i = 0                   #0부터 시작
    while i < len(lines):   #파일을 한줄씩 꺼냄
        line = lines[i].strip() #.strip() 은 줄 앞뒤의 공백이나 줄바꿈(\n)을 없애줌

        # Detect the start of a new test block  e.g. "=== TEST 42 ==="
        #re.match 는 패턴이 일치하면 결과를 반환하고, 일치하지 않으면 None 을 반환
        m = re.match(r"===\s*TEST\s+(\d+)\s*===", line) #몇번째 테스트인지 ()로 캡쳐
        # 이 줄이 === TEST N === 형태가 아니면 다음줄로 넘어가고 해당줄은 넘어가기
        if not m:
            i += 1
            continue

        #몇번째 테스트인지 숫자 꺼내기. 현재 문자열로 작성되어있기 때문에 정수형태로 바꿈
        test_id = int(m.group(1))

        # ---- Find Weight Matrix header then read SIZE rows ----
        #"Weight Matrix W" 라는 글자가 있는 줄이 나올 때까지 한 줄씩 내려가기
        while i < len(lines) and "Weight Matrix W" not in lines[i]:
            i += 1
        i += 1  # skip header line
        W = parse_matrix(lines, i, size) #지금 i 번째 줄부터 8줄을 읽어서 행렬로 만들기
        i += size

        # ---- Find Data Matrix header then read SIZE rows ----
        #"Data Matrix A" 라는 글자가 있는 줄이 나올 때까지 한 줄씩 내려가기
        while i < len(lines) and "Data Matrix A" not in lines[i]:
            i += 1
        i += 1
        A = parse_matrix(lines, i, size)     #지금 i 번째 줄부터 8줄을 읽어서 행렬로 만들기
        i += size

        # ---- Find Result Matrix header then read SIZE rows ----
        #결과값 찾아서 행렬로 만들기
        while i < len(lines) and "Result Matrix C" not in lines[i]:
            i += 1
        i += 1
        C_hw = parse_matrix(lines, i, size)
        i += size

        yield test_id, W, A, C_hw


# ──────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ──────────────────────────────────────────────────────────────────────────────

def relative_error(hw, sw):
    """
    Element-wise relative error:  |hw - ref| / max(|ref|, 1.0)
    Flooring the denominator at 1.0 avoids division-by-zero
    when the reference value is near zero.
    hw : 하드웨어 계산값
    sw : 소프트웨어 계신깂
    """
    # 두 행렬의 차이 = 원소끼리 빼기
    #sw 값이 너무 작을 때에는 1.0으로 대체
    return np.abs(hw - sw) / np.maximum(np.abs(sw), 1.0)


def format_matrix(mat, label):
    """Pretty-print a numpy matrix with a header label."""
    rows = []
    #enumerate 명령어 : 리스트를 반복할때 번호로 같이 줌.
    for i, row in enumerate(mat):
        vals = "  ".join("{:14.4f}".format(v) for v in row) # format : 문자열 안에 값 넣기.
        rows.append("  Row{}: {}".format(i, vals))
    return "{}\n{}".format(label, "\n".join(rows))


# ──────────────────────────────────────────────────────────────────────────────
# Output helper: write to both console and file at the same time
# ──────────────────────────────────────────────────────────────────────────────

def emit(out_file, text=""):
    """Print to console and append to the output file."""
    print(text)
    out_file.write(text + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
#    if len(sys.argv) < 2:
#        print("Usage: python verify_mmu.py <simulation_output.txt>")
#        sys.exit(1)

    filepath    = "./result/output.txt"
    size        = 8        # 8x8 systolic array
    threshold   = 1e-1     # flag if relative error > 0.1%
    result_path = "result/result.txt"

    total      = 0
    pass_count = 0
    fail_count = 0

    with open(result_path, "w") as out:

        emit(out, "=" * 60)
        emit(out, "  MMU_FP Simulation Output Verifier")
        emit(out, "  Input file : {}".format(filepath))
        emit(out, "  Result file: {}".format(result_path))
        emit(out, "  Tolerance  : rel_err > {:.1f}% -> FAIL".format(threshold * 100))
        emit(out, "=" * 60)

        for test_id, W, A, C_hw in parse_simulation_output(filepath, size):
            total += 1

            # Software reference: float64 matrix multiplication
            # A is (8x8) BF16 decimals, W is (8x8) BF16 decimals
            # C_ref[i][j] = sum_k  A[i][k] * W[k][j]
            C_ref = A @ W # @ 행렬 곱셈 연상

            # Compute per-element relative error
            rel_err   = relative_error(C_hw, C_ref)
            max_err   = rel_err.max()
            abs_diff  = np.abs(C_hw - C_ref)
            fail_mask = (rel_err > threshold) & (abs_diff > 1.0)

            num_wrong = int(fail_mask.sum()) # 틀린 원소의 개수


            if num_wrong == 0:
                pass_count += 1
                # Uncomment below if you also want to log passing tests:
                # emit(out, "[PASS] Test {:4d}  max_rel_err={:.2e}".format(test_id, max_err))
            else:
                fail_count += 1

                emit(out)
                emit(out, "[FAIL] Test {}  -  {}/{} elements wrong  "
                          "(max_rel_err={:.4e})".format(
                              test_id, num_wrong, size * size, max_err))

                emit(out, format_matrix(W,    "[Weight Matrix W]   (BF16 decimal from sim)"))
                emit(out, format_matrix(A,    "[Data   Matrix A]   (BF16 decimal from sim)"))
                emit(out, format_matrix(C_hw, "[HW Result  C_hw]   (FP32 decimal from sim)"))
                emit(out, format_matrix(C_ref,"[SW Reference C_ref = A @ W]  (float64)"))

                emit(out, "  [Mismatch Detail]")
                rows_idx, cols_idx = np.where(fail_mask)
                for r, c in zip(rows_idx, cols_idx):
                    emit(out,
                        "    C[{}][{}]:  HW={:14.4f}  REF={:14.4f}  rel_err={:.6f}".format(
                            r, c, C_hw[r, c], C_ref[r, c], rel_err[r, c]
                        )
                    )
                emit(out, "-" * 60)

        emit(out)
        emit(out, "=" * 60)
        emit(out, "  TOTAL  : {}".format(total))
        emit(out, "  PASS   : {}  ({:.1f}%)".format(pass_count, pass_count / total * 100 if total else 0))
        emit(out, "  FAIL   : {}  ({:.1f}%)".format(fail_count, fail_count / total * 100 if total else 0))
        emit(out, "=" * 60)

    print("\n>> Results saved to '{}'".format(result_path))


if __name__ == "__main__":
    main()